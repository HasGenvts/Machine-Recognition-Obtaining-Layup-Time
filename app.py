from flask import Flask, render_template, request, jsonify
import os
from main import BasketballDetector
from werkzeug.utils import secure_filename
import threading
from datetime import datetime
from threading import Thread

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max-limit
app.config['OUTPUT_FOLDER'] = 'static/output_clips'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}  # 允许的文件类型

# 确保上传和输出目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 全局变量
detection_results = {}  # 存储任务状态
video_clips = {}  # 存储视频片段信息

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class ProgressCallback:
    def __init__(self, task_id):
        self.task_id = task_id
        
    def update(self, frame_count, total_frames):
        """更新处理进度"""
        progress = (frame_count / total_frames) * 100
        current_status = detection_results[self.task_id]['status']
        current_clips = video_clips.get(self.task_id, [])  # 从全局变量获取clips
        
        detection_results[self.task_id].update({
            'status': current_status,
            'progress': round(progress, 1),
            'frame_count': frame_count,
            'total_frames': total_frames,
            'clips': current_clips  # 使用全局变量中的clips
        })

def save_clip_info(task_id, frame_number, clip_path):
    """保存视频片段信息到全局变量"""
    if task_id not in video_clips:
        video_clips[task_id] = []
    
    video_clips[task_id].append({
        'frame': frame_number,
        'clip_path': clip_path
    })
    print(f"已添加视频片段信息: task_id={task_id}, frame={frame_number}, path={clip_path}")

def process_video(video_path, task_id):
    try:
        # 清空该任务的视频片段信息
        if task_id in video_clips:
            video_clips[task_id] = []
        
        # 创建进度回调对象
        progress_callback = ProgressCallback(task_id)
        
        # 初始化检测器
        detector = BasketballDetector()
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # 修改BasketballDetector的save_clip方法，使其在保存视频后调用save_clip_info
        def save_clip_hook(frame_number, clip_path):
            save_clip_info(task_id, frame_number, f'output_clips/{task_id}/{os.path.basename(clip_path)}')
        
        # 将保存钩子函数添加到检测器
        detector.on_clip_saved = save_clip_hook
        
        # 调用检测方法
        goal_moments = detector.detect_goals(video_path, output_dir, progress_callback)
        
        # 更新最终状态
        detection_results[task_id].update({
            'status': 'completed',
            'clips': video_clips.get(task_id, []),  # 使用全局变量中的clips
            'message': f'检测完成，共发现 {len(video_clips.get(task_id, []))} 个进球时刻',
            'progress': 100
        })
        
        print(f"处理完成，clips: {video_clips.get(task_id, [])}")
        print(f"最终状态: {detection_results[task_id]}")
        
    except Exception as e:
        print(f"处理视频时出错: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        detection_results[task_id].update({
            'status': 'error',
            'message': f'处理失败: {str(e)}',
            'clips': video_clips.get(task_id, [])  # 保持现有的clips
        })
    finally:
        # 清理上传的文件
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"清理上传文件失败: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
        
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
        
    if not allowed_file(video_file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400
        
    try:
        # 生成任务ID
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建上传目录
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'])
        os.makedirs(upload_dir, exist_ok=True)
        
        # 保存视频文件
        video_path = os.path.join(upload_dir, f"{task_id}.mp4")
        video_file.save(video_path)
        
        # 初始化任务状态
        detection_results[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': '开始处理视频...',
            'clips': []
        }
        
        # 清空该任务的视频片段信息
        video_clips[task_id] = []
        
        # 启动处理线程
        thread = Thread(target=process_video, args=(video_path, task_id))
        thread.start()
        
        return jsonify({
            'message': '视频上传成功，开始处理',
            'task_id': task_id
        })
        
    except Exception as e:
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    """获取处理状态"""
    try:
        if task_id not in detection_results:
            return jsonify({'error': '任务不存在'}), 404
            
        status_data = detection_results[task_id].copy()
        status_data['clips'] = video_clips.get(task_id, [])  # 使用全局变量中的clips
        
        print(f"返回状态: {status_data}")
        return jsonify(status_data)
        
    except Exception as e:
        print(f"获取状态时出错: {str(e)}")
        return jsonify({'error': f'获取状态失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 