import cv2
import numpy as np
import matplotlib.pyplot as plt


# Clahe + Gamma 图像增强

# 你提供的两个增强函数
def clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):
    B, G, R = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    clahe_B = clahe.apply(B)
    clahe_G = clahe.apply(G)
    clahe_R = clahe.apply(R)
    result = cv2.merge((clahe_B, clahe_G, clahe_R))
    return result

def adjust_gamma(image, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(image, gamma_table)

# ===== 测试代码 =====
# 1. 读取原图
img = cv2.imread("/final/datasets/WUDD/images/val/16841.jpg") 
if img is None:
    raise FileNotFoundError("请确认 test.jpg 路径是否正确！")

# 转换成 RGB（matplotlib 显示需要）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Clahe 增强
clahe_img = clahe(img_rgb)

# 3. Gamma 校正（gamma<1 提亮，>1 变暗）
gamma_img = adjust_gamma(img_rgb, gamma=0.8)

# 4. 同时做 Clahe + Gamma
clahe_gamma_img = adjust_gamma(clahe_img, gamma=0.8)

# 5. 显示对比
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(clahe_img)
plt.title("Clahe Enhanced")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(gamma_img)
plt.title("Gamma Corrected")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(clahe_gamma_img)
plt.title("Clahe + Gamma Enhanced")
plt.axis("off")

plt.tight_layout()

# 保存增强后的图片
plt.savefig("./aug_result/WUDD/16841.jpg",dpi=300,bbox_inches='tight')
plt.show()
