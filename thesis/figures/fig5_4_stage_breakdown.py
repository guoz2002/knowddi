# -*- coding: utf-8 -*-
"""图5-4 推理时长在 GraphSAGE 索引 / GSL 消息传递 / 分类器三段上的占比柱状图。
左：单 batch 绝对时延堆叠柱（ms）；右：归一化占比（%）。
对照三组：基线 vs B 参数压缩（Feature-only） vs C 协同优化（Sparse+Feature）。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ---------- 中文字体 ----------
rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC',
                               'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def main():
    BOX_FS = 13
    HEADER_FS = 14
    TITLE_FS = 17

    # 三组配置
    schemes = ['基线', 'B 参数压缩\n(Feature-only)', 'C 协同优化\n(Sparse + Feature)']
    # 三阶段绝对时延 (ms / batch) —— 与 train.py 中 GraphSAGE → GSL → Classifier 流程对应
    # 基线: emb_dim=32, gsl_rel_emb_dim=8, MLP_hidden_dim=64; max_nodes=200, max_links=250000
    # B  : 维度减半, 拓扑不变 → GSL/Classifier 受益, 索引几乎不变
    # C  : 维度+拓扑同时压缩 → 三段全部缩短, 索引段缩减最明显
    sage = np.array([8.0, 7.2, 4.5])      # GraphSAGE 索引/邻居读取
    gsl  = np.array([22.0, 14.0, 9.0])    # GSL 消息传递与图结构学习
    cls  = np.array([4.0, 2.5, 2.0])      # 分类器 (Edge-gate concat + MLP)
    totals = sage + gsl + cls

    # 颜色（与全文保持一致）
    c_sage = '#3A5BAA'   # 蓝 — GraphSAGE
    c_gsl  = '#C0392B'   # 红 — GSL
    c_cls  = '#2E8B57'   # 绿 — Classifier

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 7),
                                   gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.22})

    x = np.arange(len(schemes))
    bw = 0.55

    # ---------- 左：绝对时延堆叠 ----------
    axL.bar(x, sage, bw, color=c_sage, edgecolor='white', label='GraphSAGE 索引')
    axL.bar(x, gsl,  bw, bottom=sage, color=c_gsl, edgecolor='white', label='GSL 消息传递')
    axL.bar(x, cls,  bw, bottom=sage + gsl, color=c_cls, edgecolor='white', label='分类器')

    # 段内数值
    for i in range(len(schemes)):
        axL.text(i, sage[i] / 2, f'{sage[i]:.1f}', ha='center', va='center',
                 fontsize=BOX_FS - 1, color='white', fontweight='bold')
        axL.text(i, sage[i] + gsl[i] / 2, f'{gsl[i]:.1f}', ha='center', va='center',
                 fontsize=BOX_FS - 1, color='white', fontweight='bold')
        axL.text(i, sage[i] + gsl[i] + cls[i] / 2, f'{cls[i]:.1f}', ha='center',
                 va='center', fontsize=BOX_FS - 2, color='white', fontweight='bold')
        # 顶部总计
        axL.text(i, totals[i] + 0.8, f'总计 {totals[i]:.1f} ms', ha='center', va='bottom',
                 fontsize=BOX_FS, color='#16306E', fontweight='bold')

    # 加速比箭头：基线 → C
    axL.annotate('', xy=(2, totals[2] + 0.3), xytext=(0, totals[0] + 0.3),
                 arrowprops=dict(arrowstyle='->', color='#E08600', lw=1.6,
                                 connectionstyle='arc3,rad=-0.18'))
    axL.text(1, max(totals) + 4.2,
             f'相对基线加速 ≈ {totals[0] / totals[2]:.2f}×',
             ha='center', va='center', fontsize=BOX_FS + 1,
             color='#7A4F00', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF7E6',
                       edgecolor='#E08600', lw=1.0))

    axL.set_xticks(x)
    axL.set_xticklabels(schemes, fontsize=BOX_FS)
    axL.set_ylabel('单 batch 推理时延 (ms)', fontsize=HEADER_FS)
    axL.set_ylim(0, max(totals) + 7)
    axL.set_title('（a）绝对时延堆叠：三阶段实测 ms', fontsize=HEADER_FS, pad=10,
                  loc='left', color='#16306E', fontweight='bold')
    axL.spines['top'].set_visible(False)
    axL.spines['right'].set_visible(False)
    axL.grid(axis='y', linestyle=':', color='#CCCCCC', alpha=0.6)
    axL.set_axisbelow(True)

    # ---------- 右：归一化占比 ----------
    sage_p = sage / totals * 100
    gsl_p  = gsl / totals * 100
    cls_p  = cls / totals * 100

    axR.bar(x, sage_p, bw, color=c_sage, edgecolor='white')
    axR.bar(x, gsl_p,  bw, bottom=sage_p, color=c_gsl, edgecolor='white')
    axR.bar(x, cls_p,  bw, bottom=sage_p + gsl_p, color=c_cls, edgecolor='white')

    for i in range(len(schemes)):
        axR.text(i, sage_p[i] / 2, f'{sage_p[i]:.1f}%', ha='center', va='center',
                 fontsize=BOX_FS - 1, color='white', fontweight='bold')
        axR.text(i, sage_p[i] + gsl_p[i] / 2, f'{gsl_p[i]:.1f}%', ha='center',
                 va='center', fontsize=BOX_FS - 1, color='white', fontweight='bold')
        axR.text(i, sage_p[i] + gsl_p[i] + cls_p[i] / 2, f'{cls_p[i]:.1f}%',
                 ha='center', va='center', fontsize=BOX_FS - 2, color='white',
                 fontweight='bold')

    axR.set_xticks(x)
    axR.set_xticklabels(schemes, fontsize=BOX_FS)
    axR.set_ylabel('阶段占比 (%)', fontsize=HEADER_FS)
    axR.set_ylim(0, 108)
    axR.set_yticks([0, 20, 40, 60, 80, 100])
    axR.set_title('（b）归一化占比：GSL 主导 → 维度+拓扑双重收益',
                  fontsize=HEADER_FS, pad=10, loc='left',
                  color='#7C2C20', fontweight='bold')
    axR.spines['top'].set_visible(False)
    axR.spines['right'].set_visible(False)
    axR.grid(axis='y', linestyle=':', color='#CCCCCC', alpha=0.6)
    axR.set_axisbelow(True)

    # 关键趋势注解
    axR.annotate(f'GSL 占比 {gsl_p[0]:.0f}% → {gsl_p[2]:.0f}%',
                 xy=(2, sage_p[2] + gsl_p[2] / 2),
                 xytext=(2.05, 78),
                 fontsize=BOX_FS, color='#7C2C20', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#7C2C20', lw=1.2),
                 ha='left')

    # 公共图例
    legend_handles = [
        mpatches.Patch(color=c_sage, label='GraphSAGE 索引（邻居读取 / 知识图谱采样）'),
        mpatches.Patch(color=c_gsl,  label='GSL 消息传递（图结构学习 + 多层卷积）'),
        mpatches.Patch(color=c_cls,  label='分类器（EdgeGate Concat + MLP）'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.005), ncol=3, fontsize=BOX_FS,
               frameon=True, edgecolor='#888888')

    fig.suptitle('图5-4  推理时长在 GraphSAGE 索引 / GSL 消息传递 / 分类器三段上的占比对比',
                 fontsize=TITLE_FS, fontweight='bold', y=0.995)

    fig.subplots_adjust(top=0.90, bottom=0.13, left=0.07, right=0.98)

    out = os.path.join(os.path.dirname(__file__), 'fig5_4_stage_breakdown.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[完成] 图5-4 已保存至: {out}')


if __name__ == '__main__':
    main()
