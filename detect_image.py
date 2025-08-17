import cv2
from ultralytics import YOLO
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt

# 配置路径
model_path = r'/final/REODNet/runs/train/WUDD/yolov10n-SOEP-ADown-FPSC/weights/best.pt'  # 修改为你的 YOLO 模型路径
input_folder = r'/final/datasets/WUDD/images/val'  # 修改为输入图片文件夹路径
output_folder = r'./detect_results/WUDD/yolov10n-SOEP-ADown-FPSC'  # 修改为输出结果保存的文件夹路径

# 加载训练好的 YOLO 模型
model = YOLO(model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model.to(device)

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图片
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')  # 支持的图片格式
for image_file in Path(input_folder).glob("*"):
    if image_file.suffix.lower() not in image_extensions:
        continue  # 跳过非图片文件

    # 读取图片
    image_path = str(image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        continue

    # 转换为 RGB 格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用模型进行预测
    with torch.no_grad():
        results = model.predict(image, device=device)

    # 处理检测结果
    for result in results:
        for box in result.boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # 设置不同物体边框颜色唯一，各不相同？
            # 在框上添加类别标签
            label = f"{result.names[int(box.cls)]} ({box.conf[0]:.2f})"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 保存检测结果
    output_path = os.path.join(output_folder, image_file.name)  # 使用原图片文件名保存
    cv2.imwrite(output_path, image)
    print(f"Saved detection result: {output_path}")

# 可选：显示最后一张图片的结果（用于检查效果）
plt.figure(figsize=(10, 10))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
plt.imshow(image_rgb)
plt.axis('off')  # 去掉坐标轴
plt.title("Last Detection Result")
plt.show()
