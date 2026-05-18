"""
图 5-1：DrugBank 数据集上不同子图稀疏强度下的性能与效率变化（双Y轴折线图）
运行后在当前目录生成 fig5-1_drugbank_sparse.png
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# X轴：max_nodes_per_hop，从大到小（稀疏化程度增加）
x = [10, 8, 6, 4]
macro_f1 = [91.49, 91.02, 90.51, 89.76]
acc      = [92.86, 92.74, 92.61, 92.90]
# 训练时长（秒）：3m12s=192, 2m58s=178, 2m44s=164, 2m31s=151
time_s   = [192, 178, 164, 151]
mem_mib  = [1124, 1008, 967, 935]

fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax2 = ax1.twinx()

line1, = ax1.plot(x, macro_f1, 'b-o', linewidth=2, markersize=7, label='Macro-F1 (%)')
line2, = ax1.plot(x, acc,      'g-s', linewidth=2, markersize=7, label='ACC (%)')
line3, = ax2.plot(x, mem_mib,  'r--^', linewidth=2, markersize=7, label='显存占用 (MiB)')
line4, = ax2.plot(x, time_s,   'm--D', linewidth=2, markersize=7, label='训练时长 (s)')

# 标注数值
for xi, y1, y2 in zip(x, macro_f1, acc):
    ax1.annotate(f'{y1}', xy=(xi, y1), xytext=(0, 6), textcoords='offset points',
                 ha='center', fontsize=8.5, color='blue')

ax1.axvline(x=10, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
ax1.text(10.05, 89.0, 'Baseline', fontsize=9, color='gray')
ax1.annotate('Sparse-only\n最优配置', xy=(4, 89.76),
             xytext=(5, 89.2), arrowprops=dict(arrowstyle='->', color='navy'),
             fontsize=9, color='navy')

ax1.set_xlabel('max_nodes_per_hop（子图稀疏程度 →）', fontsize=11)
ax1.set_ylabel('Score (%)', fontsize=11)
ax2.set_ylabel('资源消耗', fontsize=11, color='red')
ax1.set_ylim(88.0, 93.8)
ax2.set_ylim(100, 1400)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{v}\n(←更稀疏)' if v == 4 else str(v) for v in x])
ax1.set_title('图 5-1  DrugBank：子图稀疏强度 vs 性能/效率', fontsize=13, pad=15)

lines = [line1, line2, line3, line4]
labels_leg = [l.get_label() for l in lines]
ax1.legend(lines, labels_leg, loc='lower left', fontsize=10)
ax1.yaxis.grid(True, linestyle='--', alpha=0.4)
ax1.set_axisbelow(True)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'fig5-1_drugbank_sparse.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'[完成] 已保存: {out_path}')
plt.show()
