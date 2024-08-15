import time

import cv2
import torch
from ultralytics import YOLO
from PIL import Image
device = torch.device('mps')
# Load the trained model
model = YOLO('best.pt').to(device)

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


def draw_text_with_background(image, text, position, font_scale=0.7, font_thickness=2, bg_color=(0, 255, 0),
                              text_color=(255, 255, 255)):
    """
    在图像上绘制带有背景的文本。

    :param image: 目标图像
    :param text: 要绘制的文本
    :param position: 文本的位置 (x, y)
    :param font_scale: 字体缩放比例
    :param font_thickness: 字体粗细
    :param bg_color: 背景颜色
    :param text_color: 文本颜色
    """
    # 计算文本大小
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    # 计算背景矩形的位置
    background_start = (position[0], position[1] - text_height - 10)
    background_end = (position[0] + text_width, position[1])

    # 绘制绿色背景矩形
    cv2.rectangle(image, background_start, background_end, bg_color, cv2.FILLED)

    # 绘制文本
    cv2.putText(image, text, (position[0], position[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                font_thickness)


# 打开视频文件
cap = cv2.VideoCapture("/Users/taylor/Desktop/PythonProject/CADC/RecordVideo/08_07_14_31_58.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    # 进行目标检测
    results = model(frame)
    if len(results) > 0:
        result = results[0]
        # 绘制检测结果
        annotated_frame = result.plot()  # 获取带有检测框的图像
        # for box in result.boxes:
        #     # 提取边界框坐标、置信度和类别
        #     xyxy = box.xyxy[0].tolist()
        #     conf = box.conf[0].item()
        #     cls = int(box.cls[0].item())
        #
        #     # 绘制边界框
        #     cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        #
        #     # 绘制类别和置信度
        #     label = f"{result.names[cls]}: {int(conf*100)}%"
        #     # 在图像上绘制带背景的文本
        #     draw_text_with_background(frame, label, (int(xyxy[0]), int(xyxy[1])))

    else:
        annotated_frame = frame
    cv2.imshow('YOLOv5 Detection', annotated_frame)


    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# 释放视频捕获对象和关闭窗口
cap.release()
cv2.destroyAllWindows()





