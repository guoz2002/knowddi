"""
图 5-3：DrugBank 轻量化方案综合对比（雷达图）
运行后在当前目录生成 fig5-3_drugbank_comparison.png
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 指标：精度保留率、训练速度提升率、显存节省率（全部"越大越好"，百分比）
# Baseline 基准：Macro-F1=91.49%, 训练时长=192s, 显存=1124MiB
# Sparse-only：89.76%, 151s, 935MiB
# Feature-only：90.91%, 124s, 859MiB
# Joint：89.89%, 124s, 859MiB

categories = ['精度保留率 (%)', '训练速度\n提升率 (%)', '显存\n节省率 (%)']
N = len(categories)

def calc_scores(f1, t, mem):
    acc_retain = (1 - abs(91.49 - f1) / 91.49) * 100
    speed_gain = (1 - t / 192) * 100
    mem_save   = (1 - mem / 1124) * 100
    return [acc_retain, speed_gain, mem_save]

sparse  = calc_scores(89.76, 151, 935)
feature = calc_scores(90.91, 124, 859)
joint   = calc_scores(89.89, 124, 859)

# 雷达图角度
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# 数据闭合
def close(d): return d + d[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

ax.plot(angles, close(sparse),  'b-o', linewidth=2.5, markersize=8, label='方案A：仅拓扑稀疏化')  # 修改新增：图例改为纯中文
ax.fill(angles, close(sparse),  alpha=0.12, color='blue')

ax.plot(angles, close(feature), 'r-s', linewidth=2.5, markersize=8, label='方案B：仅特征维度压缩')  # 修改新增：图例改为纯中文
ax.fill(angles, close(feature), alpha=0.12, color='red')

ax.plot(angles, close(joint),   'g-^', linewidth=2.5, markersize=8, label='方案C：协同优化（联合）')  # 修改新增：图例改为纯中文
ax.fill(angles, close(joint),   alpha=0.12, color='green')

# 设置轴
ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=12)
ax.set_ylim(0, 25)
ax.set_yticks([5, 10, 15, 20, 25])
ax.set_yticklabels(['5%', '10%', '15%', '20%', '25%'], fontsize=8)

ax.set_title('图 5-3  DrugBank 轻量化方案综合对比', fontsize=13, pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)

# 添加具体数值标注
for val, angle, color in zip(sparse, angles[:-1], ['blue']*3):
    ax.annotate(f'{val:.1f}%', xy=(angle, val), color='blue',
                fontsize=8, ha='center')
for val, angle in zip(feature, angles[:-1]):
    ax.annotate(f'{val:.1f}%', xy=(angle, val), color='red',
                fontsize=8, ha='center')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'fig5-3_drugbank_comparison.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'[完成] 已保存: {out_path}')
print('\n各方案指标汇总（相对于Baseline的改善率）：')
print(f'  Sparse-only : 精度保留={sparse[0]:.2f}%, 速度提升={sparse[1]:.1f}%, 显存节省={sparse[2]:.1f}%')
print(f'  Feature-only: 精度保留={feature[0]:.2f}%, 速度提升={feature[1]:.1f}%, 显存节省={feature[2]:.1f}%')
print(f'  Joint       : 精度保留={joint[0]:.2f}%, 速度提升={joint[1]:.1f}%, 显存节省={joint[2]:.1f}%')
plt.show()
