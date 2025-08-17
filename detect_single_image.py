import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib import cm
import colorsys

# ----------------------------
# 配置：模型 & 单张图片路径
# ----------------------------
model_path  = r'/final/REODNet/runs/train/DUO/yolov10n-SOEP-ADown-FPSC/weights/best.pt'
image_path  = r'/final/datasets/DUO/images/test/200.jpg'    # ←改这里：单张图片
output_path = r'./detect_results/UDO/yolov10n-SOEP-ADown-FPSC/200_det.jpg'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ----------------------------
# 工具：颜色映射（每类唯一且稳定）
# ----------------------------
def _to_bgr255(rgb01):
    r, g, b = (int(255*x) for x in rgb01)
    return (b, g, r)

def hsl_to_rgb(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)

def build_class_color_map(names):
    n_cls = len(names)
    colors = {}
    base = cm.get_cmap('tab20', 20)
    for i, (cls_id, cls_name) in enumerate(names.items()):
        if i < 20:
            rgb = base(i)[:3]
            colors[cls_id] = _to_bgr255(rgb)
        else:
            h = (i * 0.61803398875) % 1.0
            s, l = 0.65, 0.55
            rgb = hsl_to_rgb(h, s, l)
            colors[cls_id] = _to_bgr255(rgb)
    return colors

def draw_label(img, text, org, color, font_scale=0.5, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)
    x, y = org
    cv2.rectangle(img, (x, y - th - baseline - 3), (x + tw + 4, y + 3), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, font_scale, (0, 0, 0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x + 2, y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# ----------------------------
# 加载模型 & 设备
# ----------------------------
model = YOLO(model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model.to(device)

# 类别名与颜色
names = model.model.names if hasattr(model, 'model') else model.names
class_colors = build_class_color_map(names)

# ----------------------------
# 单张图片推理与绘制
# ----------------------------
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f'Fail to read image: {image_path}')

with torch.no_grad():
    results = model.predict(img, device=device, verbose=False)

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = class_colors.get(cls_id, (0, 255, 0))
        thickness = max(2, int(round(0.002 * max(img.shape[:2]))))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        label = f"{names[cls_id]} {conf:.2f}"
        draw_label(img, label, (x1, y1), color, font_scale=0.5, thickness=1)

cv2.imwrite(output_path, img)
print(f"Saved: {output_path}")

# 可选：显示结果
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title("Detection Result")
# plt.show()
