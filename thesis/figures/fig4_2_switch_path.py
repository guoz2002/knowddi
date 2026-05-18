"""
图4-2 四组受控变体在 graph_structure_learner 中的开关路径示意
gsl_mode 多路选择器 → use_denoise / use_completion 布尔标志 → 边权融合公式

四组变体：
    A. baseline       use_denoise=0, use_completion=0
    B. denoise_only   use_denoise=1, use_completion=0
    C. completion_only use_denoise=0, use_completion=1
    D. full           use_denoise=1, use_completion=1

字体大小：与图4-1 风格一致（标题16 / 列标题14 / 方框12 / 公式13 / 图例12）
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig4_2_switch_path.png')


def add_box(ax, xy, w, h, text, fc='#EAF2FB', ec='#1F49D8', fontsize=12, bold=False, lw=1.4):
    box = FancyBboxPatch((xy[0], xy[1]), w, h,
                         boxstyle='round,pad=0.02,rounding_size=0.06',
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', color='#0B1F4B')


def arrow(ax, p1, p2, color='#444', lw=1.5, style='->'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style,
                                 mutation_scale=14, linewidth=lw, color=color))


def main():
    fig, ax = plt.subplots(figsize=(13, 9.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 11.5)
    ax.axis('off')

    HEADER_FS = 14
    BOX_FS = 12
    LEGEND_FS = 12

    # 标题
    ax.text(6.5, 11.05, '图4-2 四组受控变体在 graph_structure_learner 中的开关路径',
            ha='center', va='center', fontsize=16, fontweight='bold', color='#0B1F4B')

    # ── 一、命令行入口（顶层） ───────────────────
    ax.text(6.5, 10.40, '一、命令行入口  pytorch/train.py',
            fontsize=HEADER_FS, fontweight='bold', color='#0B1F4B', ha='center')

    add_box(ax, (1.5, 9.10), 10.0, 1.10,
            '--gsl_mode  ∈  {baseline ,  denoise_only ,  completion_only ,  full}\n'
            '--use_denoise (0/1)        --use_completion (0/1)        --denoise_alpha=1.0   --completion_alpha=1.0',
            fc='#FFF7E6', ec='#E08600', fontsize=BOX_FS)

    # ── 二、四组变体的开关组合（中层） ──────────
    ax.text(6.5, 8.55, '二、四组受控变体  →  gsl_mode 多路选择器分发到不同布尔组合',
            fontsize=HEADER_FS, fontweight='bold', color='#0B1F4B', ha='center')

    variants = [
        # x_left, name, use_denoise, use_completion, color_face, color_edge
        (0.30, 'A.  baseline',         '0', '0', '#F4F7FB', '#3A5BAA'),
        (3.40, 'B.  denoise_only',     '1', '0', '#EAF2FB', '#1F49D8'),
        (6.50, 'C.  completion_only',  '0', '1', '#FCEEF1', '#C0392B'),
        (9.60, 'D.  full',             '1', '1', '#EFFAEF', '#2E8B57'),
    ]
    for (x, name, ud, uc, fc, ec) in variants:
        add_box(ax, (x, 6.40), 3.10, 1.95,
                f'{name}\n\nuse_denoise = {ud}\nuse_completion = {uc}',
                fc=fc, ec=ec, fontsize=BOX_FS, bold=True)

    # 命令行 → 四组变体（虚线分发）
    for x in [1.85, 4.95, 8.05, 11.15]:
        arrow(ax, (6.5, 9.10), (x, 8.35), color='#888', lw=1.2)

    # ── 三、graph_structure_learner 内部融合公式（底层） ──
    ax.text(6.5, 5.85, '三、graph_structure_learner 中的边权融合（pytorch/model/gsl_model.py）',
            fontsize=HEADER_FS, fontweight='bold', color='#0B1F4B', ha='center')

    # 通用融合公式说明
    add_box(ax, (1.0, 4.30), 11.0, 1.20,
            r'$w_{ij}=\sigma(\,s_g + \alpha_d\cdot \mathbf{1}_{\mathrm{denoise}}\cdot s_d + \alpha_c\cdot \mathbf{1}_{\mathrm{compl}}\cdot s_c\,)$' + '\n'
            r'$\mathbf{1}_{\mathrm{denoise}}$=use_denoise，$\mathbf{1}_{\mathrm{compl}}$=use_completion；mask: $s_d$ 仅对原始边、$s_c$ 仅对新增候选边生效',
            fc='#FFF7E6', ec='#E08600', fontsize=BOX_FS + 1)

    # 四组变体在公式中的等价形式
    eq_y = 2.80
    add_box(ax, (0.30, eq_y), 3.10, 1.20,
            r'A: $w_{ij}=\sigma(s_g)$' + '\n（仅基础门控）',
            fc='#F4F7FB', ec='#3A5BAA', fontsize=BOX_FS)
    add_box(ax, (3.40, eq_y), 3.10, 1.20,
            r'B: $w_{ij}=\sigma(s_g + \alpha_d s_d)$' + '\n（去噪生效）',
            fc='#EAF2FB', ec='#1F49D8', fontsize=BOX_FS)
    add_box(ax, (6.50, eq_y), 3.10, 1.20,
            r'C: $w_{ij}=\sigma(s_g + \alpha_c s_c)$' + '\n（补全生效）',
            fc='#FCEEF1', ec='#C0392B', fontsize=BOX_FS)
    add_box(ax, (9.60, eq_y), 3.10, 1.20,
            r'D: $w_{ij}=\sigma(s_g + \alpha_d s_d + \alpha_c s_c)$' + '\n（去噪+补全联合）',
            fc='#EFFAEF', ec='#2E8B57', fontsize=BOX_FS)

    # 变体框 → 等价公式 的箭头
    for x in [1.85, 4.95, 8.05, 11.15]:
        arrow(ax, (x, 6.40), (x, 4.00), color='#888', lw=1.4, style='->')

    # 公式 → 等价形式（虚线分发）
    for x in [1.85, 4.95, 8.05, 11.15]:
        arrow(ax, (6.5, 4.30), (x, 4.00), color='#bbb', lw=0.8)

    # ── 底部说明 ─────────────────────────────────
    ax.text(6.5, 1.20,
            '注：四组变体共享同一 EdgeGateNetwork 与同一组可训练参数；\n'
            '开关只决定融合公式中各打分项的指示变量，不会引入额外参数，保证消融对比的公平性。',
            ha='center', va='center', fontsize=BOX_FS, color='#0B1F4B')

    # ── 图例 ──────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor='#FFF7E6', edgecolor='#E08600', label='命令行 / 融合公式'),
        mpatches.Patch(facecolor='#F4F7FB', edgecolor='#3A5BAA', label='A. baseline (原版退化)'),
        mpatches.Patch(facecolor='#EAF2FB', edgecolor='#1F49D8', label='B. denoise_only'),
        mpatches.Patch(facecolor='#FCEEF1', edgecolor='#C0392B', label='C. completion_only'),
        mpatches.Patch(facecolor='#EFFAEF', edgecolor='#2E8B57', label='D. full (去噪+补全)'),
    ]
    ax.legend(handles=legend_handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.04), ncol=5, frameon=False, fontsize=LEGEND_FS)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    print(f'[完成] 图4-2 已保存至: {OUT}')


if __name__ == '__main__':
    main()
