"""
图 4-1：DrugBank 数据集上四种消融变体的性能对比（分组柱状图）
运行后在当前目录生成 fig4-1_drugbank_ablation.png
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 支持中文显示
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

labels = ['Macro-F1 (%)', 'ACC (%)', "Cohen's κ (%)"]
baseline    = [91.49, 92.86, 91.53]
denoise     = [91.52, 92.66, 91.31]
completion  = [90.00, 92.89, 91.57]
full_gsl    = [90.78, 92.99, 91.68]

x = np.arange(len(labels))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 5.5))
bars1 = ax.bar(x - 1.5*width, baseline,   width, label='Baseline（无GSL）',      color='#4472C4')
bars2 = ax.bar(x - 0.5*width, denoise,    width, label='Denoise-only（仅去噪）',  color='#ED7D31')
bars3 = ax.bar(x + 0.5*width, completion, width, label='Completion-only（仅补全）', color='#70AD47')
bars4 = ax.bar(x + 1.5*width, full_gsl,   width, label='Full GSL（去噪+补全）',   color='#FF0000')

ax.set_ylim(88.5, 94.0)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('图 4-1  DrugBank 数据集消融实验性能对比', fontsize=13, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=10, loc='lower right')
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# 柱顶标注数值
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        ax.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'fig4-1_drugbank_ablation.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'[完成] 已保存: {out_path}')
plt.show()
