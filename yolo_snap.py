import cv2
import os

# 设置视频文件路径和保存目录
video_path = "/Users/taylor/Desktop/PythonProject/CADC/RecordVideo/08_12_18_13_59.mp4"
save_dir = "/Users/taylor/Desktop/PythonProject/yolov5/datasets/bucket/images"

# 确保保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
save_interval = 4  # 每隔 5 帧保存一次
saved_frame_count = 0

while True:
    # 读取视频的一帧
    ret, frame = cap.read()

    # 如果读取失败，则退出
    if not ret:
        break

    # 仅在每隔 save_interval 帧时保存图像
    if frame_count % save_interval == 0:
        # 生成保存图像的文件名
        save_path = os.path.join(save_dir, f"{saved_frame_count:04d}.jpg")
        # 保存图像
        cv2.imwrite(save_path, frame)
        print(f"Saved {save_path}")
        saved_frame_count += 1

    frame_count += 1

# 释放视频捕获对象和关闭所有 OpenCV 窗口
cap.release()
cv2.destroyAllWindows()