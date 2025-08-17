# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch

# # 原始数据：(模型名, mAP50, GFLOPs, 颜色)
# models = [
#     ('YOLOv10n', 81.5, 8.5, 'blue'),
#     ('YOLOv10n-CSFF', 83.0, 7.0, 'orangered'),
#     ('YOLOv10n-LE', 82.3, 7.4, 'purple'),
#     ('YOLOv10n-HetConv', 82.8, 8.4, 'gold'),
#     ('HCL-YOLO', 84.3, 6.1, 'green')
# ]

# # 按 GFLOPs 排序
# models_sorted = sorted(models, key=lambda x: x[2])

# # 拆分数据
# labels = [m[0] for m in models_sorted]
# mAP50_values = [m[1] for m in models_sorted]
# gflops_values = [m[2] for m in models_sorted]
# colors = [m[3] for m in models_sorted]

# # 绘图
# plt.figure(figsize=(9, 6))

# # 创建柱状图
# bars = plt.bar(range(len(labels)), mAP50_values, color=colors, edgecolor='black', width=0.5)

# # 设置坐标轴
# plt.ylabel('mAP50', fontsize=12)
# plt.xlabel('Models (GFLOPs)', fontsize=12)
# plt.ylim(80, 89)
# plt.title('Ablation experiment', fontsize=14)

# # 设置横轴刻度为模型名，旋转标签防止重叠
# plt.xticks(range(len(labels)), labels, rotation=15)

# # 在每个柱子下方标注 GFLOPs
# for i, gflop in enumerate(gflops_values):
#     plt.text(i, 79.6, f'{gflop} GFLOPs', ha='center', fontsize=9, color='dimgray')

# # 添加图例（用小矩形颜色块）
# legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(len(labels))]
# plt.legend(handles=legend_elements, loc='lower right', fontsize=9)

# # 添加子图标签 (b)
# plt.text(-0.6, 88.5, '(b)', fontsize=14, fontweight='bold')

# # 网格线
# plt.grid(axis='y', linestyle='--', alpha=0.5)

# plt.tight_layout()
# plt.savefig('/final/SA-FPSC/plot_results/GFLOPs_vs_map_barchart.png',dpi=300,bbox_inches='tight')
# plt.show()


import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 原始数据：(模型名, mAP50, GFLOPs, 颜色)
models = [
    ('YOLOv10n', 81.5, 8.5, 'blue'),
    ('YOLOv10n-CSFF', 83.0, 7.0, 'orangered'),
    ('YOLOv10n-LE', 82.3, 7.4, 'purple'),
    ('YOLOv10n-HetConv', 82.8, 8.4, 'gold'),
    ('HCL-YOLO', 84.3, 6.1, 'green')
]

# 按 GFLOPs 排序
models_sorted = sorted(models, key=lambda x: x[2])

# 拆分数据
labels = [m[0] for m in models_sorted]
mAP50_values = [m[1] for m in models_sorted]
gflops_values = [m[2] for m in models_sorted]
colors = [m[3] for m in models_sorted]

# 绘图
plt.figure(figsize=(9, 6))

# 创建柱状图
bars = plt.bar(range(len(labels)), mAP50_values, color=colors, edgecolor='black', width=0.3)

# 设置坐标轴
plt.ylabel('mAP50', fontsize=12)
plt.xlabel('GFLOPs', fontsize=12)
plt.ylim(80, 89)
plt.title('Ablation experiment', fontsize=14)

# 横坐标设置为 GFLOPs（即数值）
xtick_labels = [f'{gf} ' for gf in gflops_values]
plt.xticks(range(len(labels)), xtick_labels, rotation=0)

# 在柱子顶部添加模型名
for i, (bar, label) in enumerate(zip(bars, labels)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.3, label,
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 添加图例（小矩形颜色块）
legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(len(labels))]
plt.legend(handles=legend_elements, loc='upper right', fontsize=9)


# 网格线
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('/final/SA-FPSC/plot_results/GFLOPs_vs_map_barchart.png',dpi=300,bbox_inches='tight')
plt.show()
