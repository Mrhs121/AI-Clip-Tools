import cv2
from ultralytics import YOLO
import os
import ffmpeg
import time
from tqdm import tqdm
import subprocess
from concurrent.futures import ProcessPoolExecutor

import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# 确保当前目录下有yolov8n.pt模型文件
current_directory = os.getcwd()
model_path = os.path.join(current_directory, 'yolov8n.pt')

model = YOLO(model_path)


# 输入视频路径和输出文件夹
input_video_path = 'demo.mp4'  # 替换为您的输入视频路径
output_folder = 'output_clips/'  # 输出片段的文件夹

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 打开输入视频
cap = cv2.VideoCapture(input_video_path)

# 获取视频的帧率、宽度和高度
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 跳帧检测参数
skip_frames = int(fps)  # 每10s检测一次
print(f"skip_frames {skip_frames}, total_frames {total_frames}")

# 分段处理参数
segment_duration = 3600  # 每个分段的持续时间（秒），例如1小时
num_segments = int(total_frames / fps / segment_duration) + 1

# 初始化变量
min_clip_duration = 1  # 最小片段持续时间（秒）
tolerance_duration = 0  # 容忍时间（秒），连续未检测到人物的最长时间

# 定义函数：提取视频片段
def extract_video_segment(input_path, start_time, end_time, output_path):
    (
        ffmpeg
        .input(input_path, ss=start_time, to=end_time)
        .output(output_path, c='copy')  # 不包含音频
        .global_args('-loglevel', 'warning')
        .run(overwrite_output=True)
    )

# 定义函数：处理一个视频分段
def process_segment(segment_index):
    start_time = segment_index * segment_duration
    end_time = min((segment_index + 1) * segment_duration, total_frames / fps)
    segment_output_folder = os.path.join(output_folder, f'segment_{segment_index}')
    if not os.path.exists(segment_output_folder):
        os.makedirs(segment_output_folder)

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)  # 跳转到分段的开始时间
    frame_count = 0
    segment_start_time = None
    segment_clip_count = 0
    no_person_frame_count = 0  # 连续未检测到人物的帧数
    time_start_time = time.time()

    while cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 < end_time:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 1000 == 0:
            elapsed_time = time.time() - time_start_time
            frames_per_second = 1000 / elapsed_time
            remaining_frames = total_frames - frame_count
            estimated_time = remaining_frames / frames_per_second
            print(f"Progress [{frame_count}/{total_frames}] - Elapsed time: {elapsed_time:.2f}s - Estimated time remaining: {estimated_time:.2f}s")

        if frame_count % skip_frames == 0:
            results = model(frame, conf=0.7)
            has_person = False
            for result in results:
                for box in result.boxes:
                    if box.cls == 0:  # 类别0表示人物
                        has_person = True
                        confidence = box.conf.item()  # 获取置信度
                        cur_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        print(f"ts:{cur_time} Detected person with confidence: {confidence:.2f}")
                        break
                if has_person:
                    break

            if has_person:
                if segment_start_time is None:
                    segment_start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                no_person_frame_count = 0  # 重置未检测到人物的帧数
            else:
                cur_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                # print(f"ts:{cur_time} no_person_frame_count: {no_person_frame_count}")
                no_person_frame_count += 1
                if no_person_frame_count >= tolerance_duration * fps:
                    if segment_start_time is not None:
                        segment_end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        clip_duration = segment_end_time - segment_start_time
                        print(f"Clip duration: {clip_duration:.2f} seconds, from {segment_start_time}s to {segment_end_time}s, min_clip_duration: {min_clip_duration}")
                        if clip_duration >= min_clip_duration:
                            output_path = f"{segment_output_folder}/clip_{segment_clip_count}.mp4"
                            extract_video_segment(input_video_path, segment_start_time, segment_end_time, output_path)
                            segment_clip_count += 1
                        segment_start_time = None
                        no_person_frame_count = 0  # 重置未检测到人物的帧数

        frame_count += 1

    # 处理最后一个片段
    if segment_start_time is not None:
        segment_end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        clip_duration = segment_end_time - segment_start_time
        if clip_duration >= min_clip_duration:
            output_path = f"{segment_output_folder}/clip_{segment_clip_count}.mp4"
            print(f"Clip duration: {clip_duration:.2f} seconds, {segment_start_time} - {segment_end_time}")
            extract_video_segment(input_video_path, segment_start_time, segment_end_time, output_path)

# 使用多进程处理每个分段
def main():
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:  # 根据您的CPU线程数调整进程数
        futures = [executor.submit(process_segment, i) for i in range(num_segments)]
        for future in tqdm(futures, desc="Processing Segments"):
            future.result()

    cap.release()
    print("所有分段处理完成。")

    # 合并所有片段
    def generate_input_txt(folder_path, output_file):
        with open(output_file, 'w') as f:
            for root, dirs, files in os.walk(folder_path):
                for filename in sorted(files):
                    if filename.endswith(('.mp4', '.avi', '.mkv')):
                        video_path = os.path.join(root, filename)
                        f.write(f"file '{video_path}'\n")

    def merge_videos(input_file, output_video_path):
        command = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', input_file,
            '-c', 'copy',
            output_video_path
        ]
        subprocess.run(command, check=True)

    # 生成输入文件列表
    folder_path = output_folder
    output_file = 'input.txt'
    output_video_path = 'demo_output_video.mp4'
    generate_input_txt(folder_path, output_file)

    # 合并视频片段
    merge_videos(output_file, output_video_path)

    print(f"视频合并完成，最终视频保存在：{output_video_path}，总耗时：{time.time()-start_time}", )

if __name__ == '__main__':
    main()