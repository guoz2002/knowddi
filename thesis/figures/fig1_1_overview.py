"""
图1-1 本文两条主线工作的整体技术路线图
左：KnowDDI 主干（子图抽取 -> GraphSAGE -> GSL -> 分类器）
中：GSL 开关分支（gsl_mode -> use_denoise / use_completion -> 边权融合公式）
右：拓扑/参数双维压缩入口（max_nodes_per_hop/max_links | emb_dim/gsl_rel_emb_dim/MLP_hidden_dim）
运行：python3 fig1_1_overview.py
输出：fig1_1_overview.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 中文字体（macOS 通用）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig1_1_overview.png')


def add_box(ax, xy, w, h, text, fc='#EAF2FB', ec='#1F49D8', fontsize=10, bold=False, lw=1.4):
    box = FancyBboxPatch((xy[0], xy[1]), w, h,
                         boxstyle='round,pad=0.02,rounding_size=0.06',
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', color='#0B1F4B')


def arrow(ax, p1, p2, color='#444444', lw=1.6, style='->'):
    ar = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color)
    ax.add_patch(ar)


def dashed_arrow(ax, p1, p2, color='#1F49D8', lw=1.4):
    ar = FancyArrowPatch(p1, p2, arrowstyle='->', mutation_scale=12,
                         linewidth=lw, color=color, linestyle='--')
    ax.add_patch(ar)


def main():
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 标题
    ax.text(6.5, 7.55, '图1-1 KnowDDI 主干 + GSL 开关分支 + 拓扑/参数双维压缩入口',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#0B1F4B')

    # ── 左：KnowDDI 主干 ─────────────────────────────────
    ax.text(2.0, 6.85, '一、KnowDDI 主干', fontsize=11, fontweight='bold', color='#0B1F4B')
    backbone = [
        ((0.4, 5.6), 3.2, 0.75, '子图抽取 (DIG)\nsubgraph_extraction.py'),
        ((0.4, 4.45), 3.2, 0.75, 'GraphSAGE 全局编码\nGraphSAGE.py · pre_embed[emb_dim]'),
        ((0.4, 3.30), 3.2, 0.75, '图结构学习 GSL\ngsl_model.py · graph_structure_learner'),
        ((0.4, 2.15), 3.2, 0.75, '分类器 Classifier\nClassifier_model.py'),
    ]
    for (xy, w, h, t) in backbone:
        add_box(ax, xy, w, h, t, fc='#F4F7FB', ec='#3A5BAA', fontsize=9.5)

    # 主干箭头
    for y_top, y_bot in [(5.6, 5.20), (4.45, 4.05), (3.30, 2.90)]:
        arrow(ax, (2.0, y_top), (2.0, y_bot))

    # ── 中：GSL 开关分支 ─────────────────────────────────
    ax.text(6.5, 6.85, '二、GSL 去噪/补全开关分支', fontsize=11, fontweight='bold', color='#0B1F4B')

    add_box(ax, (4.7, 5.7), 3.6, 0.7,
            "gsl_mode ∈ {'baseline','denoise_only',\n'completion_only','full'}",
            fc='#FFF7E6', ec='#E08600', fontsize=9.5)

    add_box(ax, (4.7, 4.55), 1.7, 0.7, 'use_denoise\n(0/1)',
            fc='#EAF2FB', ec='#1F49D8', fontsize=9.5)
    add_box(ax, (6.6, 4.55), 1.7, 0.7, 'use_completion\n(0/1)',
            fc='#EAF2FB', ec='#1F49D8', fontsize=9.5)

    arrow(ax, (5.55, 5.7), (5.55, 5.25))
    arrow(ax, (7.45, 5.7), (7.45, 5.25))

    # EdgeGateNetwork
    add_box(ax, (4.5, 3.3), 4.0, 1.05,
            'EdgeGateNetwork  (gsl_model.py)\n[h_src ‖ h_dst ‖ role_emb ‖ dist ‖ rel_emb]\n→ gate_head / denoise_head / completion_head',
            fc='#EFFAEF', ec='#2E8B57', fontsize=9.5)

    arrow(ax, (5.55, 4.55), (5.55, 4.35))
    arrow(ax, (7.45, 4.55), (7.45, 4.35))

    # 融合公式
    add_box(ax, (4.5, 2.05), 4.0, 0.85,
            r'$w_{ij}=\sigma(s_g+\alpha_d\cdot \mathbf{1}_{\mathrm{denoise}}\,s_d+\alpha_c\cdot \mathbf{1}_{\mathrm{compl}}\,s_c)$',
            fc='#F4F7FB', ec='#3A5BAA', fontsize=10)

    arrow(ax, (6.5, 3.3), (6.5, 2.9))

    # 主干GSL → GSL分支（虚线）
    dashed_arrow(ax, (3.6, 3.68), (4.5, 3.85))

    # ── 右：双维压缩入口 ─────────────────────────────────
    ax.text(11.0, 6.85, '三、拓扑 / 参数 双维压缩入口', fontsize=11, fontweight='bold', color='#0B1F4B')

    # 拓扑维度
    add_box(ax, (9.2, 5.4), 3.5, 1.0,
            '拓扑维度（输入/中间张量）\n--max_nodes_per_hop : DIG BFS 每跳节点数\n--max_links : 候选完全图边数硬上限',
            fc='#FCEEF1', ec='#C0392B', fontsize=9)

    # 参数维度
    add_box(ax, (9.2, 4.05), 3.5, 1.05,
            '参数维度（容量）\n--emb_dim   (32→16)\n--gsl_rel_emb_dim (8→4)\n--MLP_hidden_dim   (64→32)',
            fc='#FCEEF1', ec='#C0392B', fontsize=9)

    # 三类方案
    add_box(ax, (9.2, 2.4), 3.5, 1.4,
            '三类轻量化方案\nB. Sparse-only  (60% 拓扑)\nC. Feature-only (50% 参数)\nD. Sparse + Feature  协同',
            fc='#FFF7E6', ec='#E08600', fontsize=9)

    # 压缩入口 → 主干（虚线，绕过中间列上下走线）
    dashed_arrow(ax, (9.2, 6.65), (3.6, 6.65))
    dashed_arrow(ax, (9.2, 4.42), (3.6, 4.42))

    # 方案 → GSL分支（点出方案D协同两路）
    arrow(ax, (10.95, 3.8), (10.95, 3.55), color='#888')
    arrow(ax, (9.2, 3.1), (8.5, 2.5), color='#888')

    # ── 底部图例 ─────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor='#F4F7FB', edgecolor='#3A5BAA', label='KnowDDI 原有模块'),
        mpatches.Patch(facecolor='#EAF2FB', edgecolor='#1F49D8', label='本文新增字段/开关'),
        mpatches.Patch(facecolor='#EFFAEF', edgecolor='#2E8B57', label='EdgeGateNetwork 内部'),
        mpatches.Patch(facecolor='#FCEEF1', edgecolor='#C0392B', label='压缩入口（命令行）'),
        mpatches.Patch(facecolor='#FFF7E6', edgecolor='#E08600', label='方案 / 模式选择'),
    ]
    ax.legend(handles=legend_handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    print(f'[完成] 图1-1 已保存至: {OUT}')


if __name__ == '__main__':
    main()
