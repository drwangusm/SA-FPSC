# msrcr.py
import cv2
import numpy as np
import os

def _safe_log(x, eps=1e-6):
    return np.log(np.maximum(x, eps))

def _gaussian_blur(img, sigma):
    # 依据 sigma 计算合适的核大小（奇数）
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

def single_scale_retinex(img, sigma):
    """
    img: float32/float64, 0~255, shape(H,W,C)
    """
    blurred = _gaussian_blur(img, sigma)
    return _safe_log(img + 1.0) - _safe_log(blurred + 1.0)

def multi_scale_retinex(img, sigmas, weights=None):
    """
    多尺度 Retinex（MSR）
    """
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)
    ret = np.zeros_like(img, dtype=np.float32)
    for w, s in zip(weights, sigmas):
        ret += w * single_scale_retinex(img, s)
    return ret

def color_restoration(img, alpha=125.0, beta=46.0):
    """
    颜色恢复项（CRF），增强色彩饱和而不过度偏色
    CRF = beta * ( log(alpha*I) - log(sum(I_channels)) )
    """
    # I: 0~255 的浮点
    I_sum = np.sum(img, axis=2, keepdims=True) + 1.0
    crf = beta * (_safe_log(alpha * img + 1.0) - _safe_log(I_sum))
    return crf

def simplest_color_balance(img, low_clip=0.01, high_clip=0.01):
    """
    最简色彩均衡（按比例裁剪直方图两端）
    img: float32, 任意范围
    """
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        # 计算低/高分位
        low_val = np.percentile(channel, low_clip * 100.0)
        high_val = np.percentile(channel, 100.0 - high_clip * 100.0)
        # 线性拉伸
        channel = np.clip((channel - low_val) / (high_val - low_val + 1e-6), 0, 1)
        out[:, :, c] = channel
    # 返回 0~255
    return (out * 255.0).astype(np.uint8)

def msrcr(
    bgr,
    sigmas=(15, 80, 250),
    weights=None,
    alpha=125.0,
    beta=46.0,
    gain=1.0,
    offset=0.0,
    low_clip=0.01,
    high_clip=0.01
):
    """
    完整 MSRCR：
    1) MSR = sum_i w_i * [ log(I) - log(Gauss(I, sigma_i)) ]
    2) CRF = beta * ( log(alpha*I) - log(sum_channels(I)) )
    3) MSRCR = gain * MSR * CRF + offset
    4) 动态范围压缩 + 简单色彩均衡
    输入: bgr (uint8) OpenCV 读取的 BGR 图像
    返回: uint8 BGR
    """
    # 转 float
    img = bgr.astype(np.float32)

    # MSR
    msr = multi_scale_retinex(img, sigmas=sigmas, weights=weights)

    # 颜色恢复
    crf = color_restoration(img, alpha=alpha, beta=beta)

    # 组合
    msrcr_val = gain * msr * crf + offset

    # 归一化到 0~1
    # 这里先按通道独立 min-max，随后做 simplest color balance 更稳定
    msrcr_min = msrcr_val.min(axis=(0, 1), keepdims=True)
    msrcr_max = msrcr_val.max(axis=(0, 1), keepdims=True)
    msrcr_norm = (msrcr_val - msrcr_min) / (msrcr_max - msrcr_min + 1e-6)

    # 最简色彩均衡（裁剪两端增强对比）
    out = simplest_color_balance(msrcr_norm.astype(np.float32), low_clip=low_clip, high_clip=high_clip)

    return out


# 读取图片（示例：BGR）
img = cv2.imread("/final/datasets/WUDD/images/val/16841.jpg")
if img is None:
    raise FileNotFoundError("找不到 test.jpg，请检查路径！")

# 参数推荐（经典论文/工程常用）
params = dict(
    sigmas=(15, 80, 250),   # 多尺度
    alpha=125.0,
    beta=46.0,
    gain=1.0,
    offset=0.0,
    low_clip=0.01,
    high_clip=0.01,
)

out = msrcr(img, **params)

img = cv2.imread("/final/datasets/WUDD/images/val/16841.jpg")
if img is None:
    raise FileNotFoundError("找不到 /final/datasets/WUDD/images/val/16841.jpg，请检查路径！")

params = dict(sigmas=(15,80,250), alpha=125.0, beta=46.0, gain=1.0, offset=0.0,
              low_clip=0.01, high_clip=0.01)
out = msrcr(img, **params)

save_dir = "./aug_result/WUDD"
os.makedirs(save_dir, exist_ok=True)               # ① 确保目录存在
save_path = os.path.join(save_dir, "msrcr_16841.jpg")

ok = cv2.imwrite(save_path, out)                   # ② 直接保存 BGR 图像
if not ok:
    raise RuntimeError(f"保存失败：{save_path}")
print(f"已保存：{save_path}")


