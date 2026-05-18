"""
图 5-2：DrugBank 数据集上不同特征压缩强度下的性能与效率变化（双Y轴折线图）
运行后在当前目录生成 fig5-2_drugbank_feature.png
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# X轴：emb_dim（嵌入维度），从大到小（压缩程度增加）
x = [32, 24, 16, 8]
macro_f1 = [91.49, 91.15, 90.91, 89.43]
acc      = [92.86, 92.77, 92.80, 92.48]
# 训练时长（秒）：3m12s=192, 2m31s=151, 2m04s=124, 1m52s=112
time_s   = [192, 151, 124, 112]
mem_mib  = [1124, 956, 859, 782]

fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax2 = ax1.twinx()

line1, = ax1.plot(x, macro_f1, 'b-o', linewidth=2, markersize=7, label='Macro-F1 (%)')
line2, = ax1.plot(x, acc,      'g-s', linewidth=2, markersize=7, label='ACC (%)')
line3, = ax2.plot(x, mem_mib,  'r--^', linewidth=2, markersize=7, label='显存占用 (MiB)')
line4, = ax2.plot(x, time_s,   'm--D', linewidth=2, markersize=7, label='训练时长 (s)')

# 标注最优配置
ax1.annotate('Feature-only 最优\n90.91%', xy=(16, 90.91),
             xytext=(19, 89.6), arrowprops=dict(arrowstyle='->', color='navy'),
             fontsize=9, color='navy')

ax1.axvline(x=32, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
ax1.text(31.2, 89.0, 'Baseline', fontsize=9, color='gray', ha='right')

ax1.set_xlabel('emb_dim（特征压缩程度 →）', fontsize=11)
ax1.set_ylabel('Score (%)', fontsize=11)
ax2.set_ylabel('资源消耗', fontsize=11, color='red')
ax1.set_ylim(88.0, 93.8)
ax2.set_ylim(500, 1400)
ax1.set_xticks(x)
ax1.set_title('图 5-2  DrugBank：特征压缩强度 vs 性能/效率', fontsize=13, pad=15)

lines = [line1, line2, line3, line4]
labels_leg = [l.get_label() for l in lines]
ax1.legend(lines, labels_leg, loc='lower left', fontsize=10)
ax1.yaxis.grid(True, linestyle='--', alpha=0.4)
ax1.set_axisbelow(True)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'fig5-2_drugbank_feature.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'[完成] 已保存: {out_path}')
plt.show()
