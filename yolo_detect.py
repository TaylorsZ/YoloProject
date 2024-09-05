import os
import time

import cv2
import torch
from ultralytics import YOLO
from PIL import Image
device = torch.device('mps')
# Load the trained model
model = YOLO('best_0827.pt').to(device)

# Load a test image
# image_path = 'datasets/bucket/images/0000.jpg'
# image = cv2.imread(image_path)

# Perform inference
# cu = time.time()
# results = model(image)
# print("耗时：", int((time.time() - cu)*1000), "ms")
# print(f"识别到{len(results)}个")
# if len(results) > 0:
#     result = results[0]
#
# for result in results:
#     for box in result.boxes:
#         cv2.rectangle(image, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
#                       (0, 255, 0), 2)
#         cv2.imshow('YOLOv5 Detection', image)
#         cv2.waitKey(0)

# 指定文件夹路径
folder_path = '/Users/taylor/Desktop/PythonProject/CADC/RecordVideo'

def get_mp4_files(directory):
    # 列出目录下的所有文件和文件夹
    files = os.listdir(directory)
    # 筛选出所有.mp4文件
    mp4_files = [file for file in files if file.endswith('.mp4')]
    return mp4_files
file_names = get_mp4_files(folder_path)
def format_time(seconds):
    """将秒数格式化为 'MM:SS' 格式"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"
def play_video(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(2)
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 计算视频的总时长
    total_duration = total_frames / fps
    # 设定快进和快退步长（帧数）
    skip_frames = int(fps * 3)  # 3秒的帧数

    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频")
            break
            # 计算当前播放时间
        current_time = current_frame / fps
        # 格式化当前时间和总时间
        time_text = f"{format_time(current_time)} / {format_time(total_duration)}"
        # 在视频帧上显示时间信息
        cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 进行目标检测
        results = model(frame, conf=0.8)
        if len(results) > 0:
            result = results[0]
            # 绘制检测结果
            annotated_frame = result.plot()  # 获取带有检测框的图像
        else:
            print("未检测到目标")
            annotated_frame = frame
        cv2.imshow(f'Detect Bucket-{video_path}', annotated_frame)
        key = cv2.waitKey(20)  # 等待 20ms 来显示帧，按键操作可捕获
        if key == 27:  # 按 'ESC' 键退出
            break
        elif key == 3:  # 按右箭头键快进
            current_frame += skip_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        elif key == 2:  # 按左箭头键快退
            current_frame = max(0, current_frame - skip_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        else:
            current_frame += 1

    # 释放视频捕获对象和关闭窗口
    cap.release()
    cv2.destroyAllWindows()
# url = "/Users/taylor/Documents/QGroundControl/Video/2024-08-23_17.17.11.mkv"
# play_video(url)
play_video(f"{folder_path}/{file_names[3]}")
for file_name in file_names:
    play_video(f"{folder_path}/{file_name}")





