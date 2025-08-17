import os
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib import cm

# ----------------------------
# 配置路径
# ----------------------------
model_path = r'/final/REODNet/runs/train/DUO/yolov10n-SOEP-ADown-FPSC/weights/best.pt'
input_folder = r'/final/datasets/WUDD/images/val'
output_folder = r'./detect_results/UDO/yolov10n-SOEP-ADown-FPSC'

# ----------------------------
# 工具：颜色映射（每类唯一且稳定）
# ----------------------------
def _to_bgr255(rgb01):
    """将(0~1)的RGB转为OpenCV用的BGR 0~255整数"""
    r, g, b = (int(255*x) for x in rgb01)
    return (b, g, r)

def build_class_color_map(names):
    """
    给每个类别名分配一个稳定的颜色（BGR, 0~255）
    优先使用tab20分配20种高区分度颜色；类别>20时再用分段HSL拓展。
    """
    n_cls = len(names)
    colors = {}

    # 1) 先用 matplotlib 的 tab20（20 个可区分颜色）
    base = cm.get_cmap('tab20', 20)  # 返回 20 个 (r,g,b,a), 0~1
    for i, (cls_id, cls_name) in enumerate(names.items()):
        if i < 20:
            rgb = base(i)[:3]
            colors[cls_id] = _to_bgr255(rgb)
        else:
            # 2) 超过20个，再用HSL分布色相（避免太接近）
            # 均匀分布色相，固定高亮与中等饱和度
            h = (i * 0.61803398875) % 1.0  # 黄金分割打散
            s = 0.65
            l = 0.55
            # HSL->RGB
            rgb = hsl_to_rgb(h, s, l)
            colors[cls_id] = _to_bgr255(rgb)

    return colors

def hsl_to_rgb(h, s, l):
    """把 HSL(0~1) 转 RGB(0~1)"""
    # python 自带 colorsys 也可以：colorsys.hls_to_rgb(h, l, s)
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)

# 绘制带底框的文本（清晰）
def draw_label(img, text, org, color, font_scale=0.5, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)
    x, y = org
    # 背景条
    cv2.rectangle(img, (x, y - th - baseline - 3), (x + tw + 4, y + 3), color, -1)
    # 文字用黑/白描边以增强对比
    cv2.putText(img, text, (x + 2, y - 2), font, font_scale, (0, 0, 0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x + 2, y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# ----------------------------
# 加载模型 & 设备
# ----------------------------
model = YOLO(model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model.to(device)

# 类别名字典（id->name），并构建颜色映射
names = model.model.names if hasattr(model, 'model') else model.names
class_colors = build_class_color_map(names)

# 输出目录
os.makedirs(output_folder, exist_ok=True)

# 支持的图片格式
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# ----------------------------
# 推理与绘制
# ----------------------------
last_image_for_show = None

for image_file in Path(input_folder).glob("*"):
    if image_file.suffix.lower() not in image_extensions:
        continue

    image = cv2.imread(str(image_file))
    if image is None:
        print(f"Failed to read image: {image_file}")
        continue

    with torch.no_grad():
        results = model.predict(image, device=device, verbose=False)

    # 逐张图绘制
    for result in results:
        # Ultralytics 的坐标在 result.boxes
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = class_colors.get(cls_id, (0, 255, 0))  # 若找不到就用绿色备用
            thickness = max(2, int(round(0.002 * max(image.shape[:2]))))  # 随分辨率自适应

            # 框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            # 标签
            label = f"{names[cls_id]} {conf:.2f}"
            draw_label(image, label, (x1, y1), color, font_scale=0.5, thickness=1)

    # 保存
    out_path = os.path.join(output_folder, image_file.name)
    cv2.imwrite(out_path, image)
    print(f"Saved: {out_path}")
    last_image_for_show = image

# 可选：展示最后一张
if last_image_for_show is not None:
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(last_image_for_show, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Last Detection Result")
    plt.show()
