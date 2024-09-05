import torch
from ultralytics import YOLO

print("MPS is available:", torch.backends.mps.is_available())
# Check if MPS is available and set the device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load the model
model = YOLO('yolov8n.pt').to(device)
'''
	1.	task: 指定任务类型，如 detect（目标检测）、segment（实例分割）、classify（图像分类）。你已经在使用这个参数。
	2.	data: 用于指定数据集配置文件的路径。你已经在使用这个参数。
	3.	epochs: 训练的总周期数。你已经在使用这个参数。
	4.	imgsz: 指定输入图像的尺寸（宽和高），如 640。你已经在使用这个参数。
	5.	batch: 每个训练周期中使用的批次大小（Batch Size）。通常设置为 16、32、64 等值，具体数值取决于 GPU 的内存容量。例如：batch=16。
	6.	lr0: 初始学习率。用于控制训练过程中模型权重更新的步长。例如：lr0=0.01。
	7.	lrf: 最终学习率的系数。学习率将在训练期间从初始值 lr0 逐渐降低到 lr0*lrf。例如：lrf=0.01。
	8.	momentum: 优化器中的动量因子，用于控制梯度更新的惯性。典型值为 0.937。
	9.	weight_decay: 权重衰减（L2 正则化）系数，防止过拟合。例如：weight_decay=0.0005。
	10.	optimizer: 指定使用的优化器。可以选择 SGD 或 Adam 等。例如：optimizer='SGD'。
	11.	device: 指定训练的设备，例如 cpu、cuda:0、0,1,2,3（多 GPU）。如果不指定，YOLOv8 会自动选择可用的设备。例如：device='cuda:0'。
	12.	workers: 数据加载时使用的进程数量。这个参数可以加速数据的加载过程。典型值为 8 或 16。例如：workers=8。
	13.	project: 指定训练结果保存的项目目录。例如：project='runs/train'。
	14.	name: 训练运行的名称，用于创建保存结果的子目录。例如：name='exp'。
	15.	exist_ok: 如果为 True，表示允许在已经存在的目录中继续保存结果，而不会报错。例如：exist_ok=True。
	16.	pretrained: 是否使用预训练模型。可以设置为 True 或指定预训练模型的路径。
	17.	patience: 提前停止的周期数。如果模型在指定周期数内没有改进，将提前停止训练。默认值为 100。
	18.	freeze: 冻结模型的某些层，通常用于迁移学习。接受整数或列表。例如：freeze=10（冻结前 10 层）。
	19.	label_smoothing: 标签平滑系数，用于减小过拟合。典型值为 0.0 到 0.1。
	20.	save_period: 模型保存的周期间隔。例如：save_period=10 表示每 10 个周期保存一次模型。
	21.	val: 验证数据的比例或路径。可以用来指定训练过程中使用多少数据进行验证。
	22.	augment: 数据增强。设置为 True 时，训练过程中会自动进行数据增强。
	23.	verbose: 设置为 True 时，将输出详细的训练日志。
	24.	seed: 随机种子。用于保证训练的可重复性。
'''
# Training setup
model.train(task ="detect",data='datasets/bucket.yaml', epochs=5000, imgsz=640)