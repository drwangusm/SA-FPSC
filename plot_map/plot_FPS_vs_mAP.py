import matplotlib.pyplot as plt

# 各模型的数据点（mAP50, FPS, color, size, label）
data = [
    (81, 20, 'yellow', 800, 'Faster R-CNN'),
    (63, 40, 'gray', 1000, 'RetinaNet'),
    (73, 30, 'magenta', 600, 'SSD'),
    (75, 160, 'lime', 200, 'YOLOv5n'),
    (82.5, 100, 'green', 200, 'YOLOv7-tiny'),
    (83.5, 110, 'black', 200, 'YOLOv8n'),
    (84.5, 180, 'red', 200, 'YOLOv9t'),
    (83, 140, 'blue', 200, 'YOLOv10n'),
    (84, 120, 'orange', 200, 'YOLOv11n'),
    (85.5, 160, 'pink', 200, 'Ours')
]

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制散点图
for x, y, color, size, label in data:
    plt.scatter(x, y, s=size, c=color, label=label, alpha=0.8, edgecolors='black')

# 设置标题与坐标轴
plt.title('Performance Comparison', fontsize=14)
plt.xlabel('mAP50', fontsize=12)
plt.ylabel('FPS', fontsize=12)
plt.xlim(60, 90)
plt.ylim(0, 200)

# 去重图例（防止重复）
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9)

# 网格和显示
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/final/SA-FPSC/plot_results/fps_vs_map.png',dpi=300,bbox_inches='tight')
plt.show()
