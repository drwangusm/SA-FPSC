# import matplotlib.pyplot as plt

# # 设置字体支持 (可选，防止中文乱码或加粗标题)
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['axes.unicode_minus'] = False

# # 定义数据：x = GFLOPs, y = mAP50, color, size, label
# data = [
#     (8.5, 81.5, 'blue', 900, 'YOLOv10n'),
#     (7.0, 83.0, 'orangered', 800, 'YOLOv10n-CSFF'),
#     (7.4, 82.3, 'purple', 800, 'YOLOv10n-LE'),
#     (8.4, 82.8, 'gold', 900, 'YOLOv10n-HetConv'),
#     (6.1, 84.3, 'green', 900, 'HCL-YOLO')
# ]

# # 创建图像
# plt.figure(figsize=(7, 5))

# # 绘制散点图
# for x, y, color, size, label in data:
#     plt.scatter(x, y, s=size, c=color, label=label, alpha=0.9, edgecolors='black')

# # 设置标题与坐标轴标签
# plt.title('Ablation experiment', fontsize=13)
# plt.xlabel('GFLOPs', fontsize=12)
# plt.ylabel('mAP50', fontsize=12)
# plt.xlim(5.8, 8.8)
# plt.ylim(80, 89)

# # 添加图例
# plt.legend(title='', fontsize=9, loc='best')

# # 添加子图标签 (b)
# plt.text(5.8, 88.5, '(b)', fontsize=14, fontweight='bold')

# # 设置网格线
# plt.grid(True, linestyle='--', alpha=0.5)

# # 显示图像
# plt.tight_layout()
# plt.savefig('/final/SA-FPSC/plot_results/GFLOPs_vs_map.png',dpi=300,bbox_inches='tight')
# plt.show()


import matplotlib.pyplot as plt

# 原始散点数据（保持原来大小不变）
data = [
    (8.5, 81.5, 'blue', 900, 'YOLOv10n'),
    (7.0, 83.0, 'orangered', 800, 'YOLOv10n-CSFF'),
    (7.4, 82.3, 'purple', 800, 'YOLOv10n-LE'),
    (8.4, 82.8, 'gold', 900, 'YOLOv10n-HetConv'),
    (6.1, 84.3, 'green', 900, 'HCL-YOLO')
]

# 创建图形
plt.figure(figsize=(7, 5))

# 用于自定义 legend 句柄
scatter_handles = []

# 绘制散点，并收集 legend 句柄（单独指定 legend marker size）
for x, y, color, size, label in data:
    sc = plt.scatter(x, y, s=size, c=color, alpha=0.9, edgecolors='black')
    scatter_handles.append(plt.Line2D([], [], linestyle='', marker='o', color=color, markersize=8, label=label))

# 坐标轴设置
plt.title('Ablation experiment', fontsize=13)
plt.xlabel('GFLOPs', fontsize=12)
plt.ylabel('mAP50', fontsize=12)
plt.xlim(5.8, 8.8)
plt.ylim(80, 89)

# 图例：使用自定义 legend 句柄来减小圆圈
plt.legend(handles=scatter_handles, fontsize=9, loc='best')

# 添加子图标题 (b)
plt.text(5.8, 88.5, '(b)', fontsize=14, fontweight='bold')

# 网格和布局
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/final/SA-FPSC/plot_results/GFLOPs_vs_map.png',dpi=300,bbox_inches='tight')
plt.show()

