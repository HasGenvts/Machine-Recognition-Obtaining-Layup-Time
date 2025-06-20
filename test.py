import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import subprocess
from collections import deque

class BasketballDetector:
    def __init__(self, show_inference=False):
        """
        初始化篮球检测器
        :param show_inference: 是否显示推理过程界面，默认False
        """
        print("正在加载YOLOv8n模型...")
        self.model = YOLO('yolov8n.pt')
        print("模型加载完成！")
        
        # 显示配置
        self.show_inference = show_inference
        
        # 存储球的轨迹
        self.ball_positions = deque(maxlen=30)  # 存储1秒的轨迹
        self.velocities = deque(maxlen=10)      # 存储速度历史
        self.directions = deque(maxlen=10)      # 存储方向历史
        
        # 进球判定参数
        self.last_goal_time = 0
        self.min_frames_gap = 30  # 两次进球最小间隔帧数
        self.potential_goal = False  # 潜在进球标记
        self.down_count = 0  # 连续下落计数
        self.potential_goal_start = 0  # 记录潜在进球开始的帧
        
        # 存储人的头部位置
        self.head_positions = deque(maxlen=5)  # 存储最近5帧的头部位置
        
        # 帧缓冲区（改用字典存储，键为帧号）
        self.frame_buffer = {}
        self.max_buffer_frames = 240  # 存储8秒的帧 (假设30fps)
        self.buffer_start_frame = 0   # 缓冲区起始帧号
        
        # 记录处理开始时间
        self.process_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 显示参数
        self.colors = {
            'ball': (0, 255, 0),     # 绿色
            'trajectory': (255, 255, 0),  # 黄色
            'velocity': (0, 165, 255),    # 橙色
            'person': (255, 0, 0)     # 红色
        }
        
    def analyze_trajectory(self):
        """分析球的运动轨迹"""
        if len(self.ball_positions) < 6:
            return None, None, None
            
        # 获取最近的点
        points = list(self.ball_positions)
        
        # 计算最近5帧的运动
        recent_points = points[-6:]
        velocities = []
        directions = []
        
        # 计算速度和方向
        for i in range(len(recent_points)-1):
            p1 = recent_points[i]
            p2 = recent_points[i+1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # 速度（像素/帧）
            velocity = np.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
            
            # 方向（弧度）
            direction = np.arctan2(dy, dx)
            directions.append(direction)
        
        # 保存到历史记录
        if velocities:
            self.velocities.append(np.mean(velocities))
        if directions:
            self.directions.append(np.mean(directions))
        
        # 判断是否在下落
        is_falling = False
        if len(points) >= 3:
            recent_y = [p[1] for p in points[-3:]]
            is_falling = all(recent_y[i] < recent_y[i+1] for i in range(len(recent_y)-1))
        
        # 获取当前球的高度（y坐标）
        current_height = points[-1][1] if points else None
        
        return current_height, is_falling, np.mean(velocities) if velocities else 0
        
    def draw_detection(self, frame):
        """在帧上绘制检测结果"""
        # 绘制球的轨迹
        if len(self.ball_positions) >= 2:
            points = np.array(list(self.ball_positions), dtype=np.int32)
            cv2.polylines(frame, [points], False, self.colors['trajectory'], 2)
        
        # 绘制当前球的位置
        if self.ball_positions:
            last_pos = self.ball_positions[-1]
            cv2.circle(frame, last_pos, 5, self.colors['ball'], -1)
            cv2.circle(frame, last_pos, 15, self.colors['ball'], 2)
            
            # 绘制速度方向
            if len(self.velocities) > 0 and len(self.directions) > 0:
                velocity = self.velocities[-1]
                direction = self.directions[-1]
                end_point = (
                    int(last_pos[0] + velocity * np.cos(direction)),
                    int(last_pos[1] + velocity * np.sin(direction))
                )
                cv2.arrowedLine(frame, last_pos, end_point, self.colors['velocity'], 2)
        
        return frame
        
    def update_frame_buffer(self, frame, frame_count):
        """更新帧缓冲区"""
        # 存储当前帧
        self.frame_buffer[frame_count] = frame.copy()
        
        # 如果缓冲区过大，删除旧帧
        while len(self.frame_buffer) > self.max_buffer_frames:
            min_frame = min(self.frame_buffer.keys())
            del self.frame_buffer[min_frame]
            self.buffer_start_frame = min(self.frame_buffer.keys())
            
    def save_clip(self, video_path, goal_frame, fps, output_dir):
        """保存进球片段"""
        try:
            # 创建基础输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 使用处理开始时间作为子文件夹名
            sub_dir = os.path.join(output_dir, self.process_start_time)
            os.makedirs(sub_dir, exist_ok=True)
            
            # 生成输出文件路径，使用帧号作为文件名
            output_path = os.path.join(sub_dir, f'shot_{goal_frame:06d}.mp4')
            temp_path = output_path + "_temp.mp4"
            
            # 获取视频尺寸
            first_frame = next(iter(self.frame_buffer.values()))
            height, width = first_frame.shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            # 计算要保存的帧范围
            # 由于进球判定可能有延迟，我们多保存一些前面的帧
            frames_before = int(fps * 4)  # 前4秒
            frames_after = int(fps * 2)   # 后2秒
            
            start_frame = max(self.buffer_start_frame, goal_frame - frames_before)
            end_frame = min(max(self.frame_buffer.keys()), goal_frame + frames_after)
            
            print(f"进球时刻: 第{goal_frame}帧")
            print(f"缓冲区范围: {self.buffer_start_frame} - {max(self.frame_buffer.keys())}")
            print(f"实际截取范围: {start_frame} - {end_frame}")
            print(f"预计进球时刻在视频片段的第{frames_before/fps:.1f}秒处")
            print(f"视频将保存到: {output_path}")
            
            # 按顺序写入帧
            frames_written = 0
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx in self.frame_buffer:
                    out.write(self.frame_buffer[frame_idx])
                    frames_written += 1
            
            out.release()
            print(f"已写入 {frames_written} 帧到临时文件")
            
            if frames_written == 0:
                print("错误：没有写入任何帧")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return
            
            # 检查临时文件是否存在且大小正常
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                print("错误：临时文件创建失败或为空")
                return
                
            try:
                # 使用ffmpeg进行转换，添加更多参数确保兼容性
                command = [
                    'ffmpeg', '-y',
                    '-i', temp_path,
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-pix_fmt', 'yuv420p',  # 确保输出格式兼容
                    '-movflags', '+faststart',  # 优化网络播放
                    '-profile:v', 'baseline',  # 使用基础配置文件
                    '-level', '3.0',  # 设置编码级别
                    output_path
                ]
                
                # 执行ffmpeg命令并捕获输出
                result = subprocess.run(command, 
                                     capture_output=True, 
                                     text=True)
                
                if result.returncode != 0:
                    print(f"FFmpeg转换失败，错误码：{result.returncode}")
                    print(f"错误输出：{result.stderr}")
                else:
                    print("FFmpeg转换成功")
                    
            except Exception as e:
                print(f"FFmpeg执行出错: {str(e)}")
            finally:
                # 清理临时文件
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        print(f"删除临时文件失败: {str(e)}")
            
        except Exception as e:
            print(f"保存视频片段失败: {str(e)}")
            import traceback
            print(f"错误详情:\n{traceback.format_exc()}")
        
    def get_head_position(self, results, frame):
        """获取画面中人的头部位置"""
        head_y = None
        for r in results.boxes.data:
            x1, y1, x2, y2, conf, cls = r
            cls = int(cls)
            if cls == 0 and conf > 0.3:  # 人的类别是0
                # 取边界框上方1/6作为头部位置
                head_y = int(y1 + (y2 - y1) / 6)
                if self.show_inference:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                self.colors['person'], 2)
                    cv2.circle(frame, (int((x1 + x2) / 2), head_y), 5, self.colors['person'], -1)
                break
        
        if head_y is not None:
            self.head_positions.append(head_y)
            # 使用最近几帧的平均值来平滑头部位置
            return int(sum(self.head_positions) / len(self.head_positions))
        elif len(self.head_positions) > 0:
            # 如果当前帧没检测到，但之前有检测到，使用最近的位置
            return self.head_positions[-1]
        return None
        
    def is_goal(self, ball_pos, frame_count, head_y):
        """基于球相对于头部的高度判断投篮过程"""
        if not ball_pos or head_y is None:
            return False, None
            
        # 分析轨迹
        current_height, is_falling, velocity = self.analyze_trajectory()
        
        if current_height is None:
            return False, None
            
        # 判断球是否高于头部
        head_offset = 50  # 球需要高于头部多少像素才算投篮
        
        # 如果球高于头部且还没有标记为潜在进球
        if current_height < (head_y - head_offset) and not self.potential_goal:
            self.potential_goal = True
            self.potential_goal_start = frame_count
            self.down_count = 0
            print(f"检测到投篮开始，帧号: {frame_count}, 球的高度: {current_height}, 头部位置: {head_y}")
            return False, None
            
        if self.potential_goal:
            # 更新下落计数
            if is_falling:
                self.down_count += 1
            else:
                self.down_count = max(0, self.down_count - 1)
            
            # 投篮结束条件：
            # 1. 球低于头部
            # 2. 有连续的下落过程
            if (current_height > head_y and  # 球低于头部
                self.down_count >= 2 and  # 连续2帧下落
                velocity < 100):  # 速度合理
                
                # 投篮结束，记录时间点
                actual_goal_frame = frame_count
                
                # 重置状态
                self.potential_goal = False
                self.down_count = 0
                print(f"检测到投篮结束，帧号: {frame_count}, 球的高度: {current_height}, 头部位置: {head_y}")
                
                return True, self.potential_goal_start
                
        return False, None
        
    def detect_goals(self, video_path, output_dir='output_clips'):
        """检测视频中的进球"""
        print(f"开始处理视频: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开视频文件")
            return
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        goals_detected = []
        
        print(f"视频信息: {total_frames} 帧, {fps} FPS")
        
        # 根据fps调整缓冲区大小
        self.max_buffer_frames = fps * 8  # 存储8秒的帧
        
        # 如果需要显示推理界面，创建窗口
        if self.show_inference:
            cv2.namedWindow('Basketball Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Basketball Detection', 1280, 720)
            print("已创建显示窗口")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # 更新帧缓冲区
                self.update_frame_buffer(frame, frame_count)
                
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"处理进度: {progress:.1f}%")
                
                # 使用YOLO检测物体
                results = self.model(frame, verbose=False)[0]
                
                # 获取头部位置
                head_y = self.get_head_position(results, frame)
                
                # 找到篮球
                ball_detected = False
                current_ball_pos = None
                
                for r in results.boxes.data:
                    x1, y1, x2, y2, conf, cls = r
                    cls = int(cls)
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    
                    if cls == 32 and conf > 0.2:  # 篮球
                        ball_detected = True
                        current_ball_pos = center
                        self.ball_positions.append(center)
                        
                        if self.show_inference:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                        self.colors['ball'], 2)
                
                if self.show_inference:
                    frame = self.draw_detection(frame)
                    
                    # 显示头部位置
                    if head_y is not None:
                        cv2.line(frame, (0, head_y), (frame.shape[1], head_y), 
                                self.colors['person'], 1)
                    
                    status_text = [
                        f"Frame: {frame_count}/{total_frames}",
                        f"Ball Height: {current_ball_pos[1] if current_ball_pos else 'N/A'}",
                        f"Head Height: {head_y if head_y is not None else 'N/A'}",
                        f"Potential Shot: {'Yes' if self.potential_goal else 'No'}",
                        f"Down Count: {self.down_count}"
                    ]
                    for i, text in enumerate(status_text):
                        cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (255, 255, 255), 2)
                
                if (ball_detected and head_y is not None and
                    frame_count - self.last_goal_time > self.min_frames_gap):
                    
                    is_goal, shot_start_frame = self.is_goal(current_ball_pos, frame_count, head_y)
                    if is_goal:
                        print(f"检测到完整投篮过程！开始帧: {shot_start_frame}")
                        goals_detected.append(shot_start_frame)
                        self.save_clip(video_path, shot_start_frame, fps, output_dir)
                        self.last_goal_time = frame_count
                        
                        if self.show_inference:
                            cv2.putText(frame, "Shot Detected!", (frame.shape[1]//2 - 100, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                
                if self.show_inference:
                    cv2.imshow('Basketball Detection', frame)
                    wait_time = max(1, int(1000/fps))
                    key = cv2.waitKey(wait_time)
                    if key & 0xFF == ord('q'):
                        print("用户按下Q键，退出处理")
                        break
                    elif key & 0xFF == ord('p'):
                        print("用户按下P键，暂停处理")
                        while True:
                            if cv2.waitKey(100) & 0xFF == ord('p'):
                                print("继续处理")
                                break
                        
        except Exception as e:
            print(f"处理视频时出错: {str(e)}")
        finally:
            cap.release()
            if self.show_inference:
                cv2.destroyAllWindows()
                print("已关闭显示窗口")
            
        print(f"视频处理完成，共检测到 {len(goals_detected)} 个进球")
        return goals_detected

def main():
    # 使用示例
    detector = BasketballDetector(show_inference=True)  # 默认不显示推理界面
    video_path = "D:\\Project\\work\\20250621-py\\videos\\test.mp4"  # 替换为实际的视频路径
    detector.detect_goals(video_path)

if __name__ == "__main__":
    main()
