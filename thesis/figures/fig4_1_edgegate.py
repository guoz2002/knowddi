"""
图4-1 EdgeGateNetwork 内部结构示意图
画布尺寸保持原始 (13, 8.5)，xlim 不变；纵向 ylim 扩展以容纳更大字号；
方框宽度与原始一致，仅在需要时纵向加高。
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig4_1_edgegate.png')


def add_box(ax, xy, w, h, text, fc='#EAF2FB', ec='#1F49D8', fontsize=11, bold=False, lw=1.4):
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
    # 画布尺寸保持原始；xlim 保持 0~13；ylim 仅做必要的纵向扩展
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 11.5)
    ax.axis('off')

    HEADER_FS = 14
    BOX_FS = 12
    LEGEND_FS = 12

    # 标题
    ax.text(6.5, 11.10, '图4-1 EdgeGateNetwork 内部接线（输入特征 → Encoder → 三路输出头 → 边权融合）',
            ha='center', va='center', fontsize=16, fontweight='bold', color='#0B1F4B')

    # ── 1. 输入特征列（左）─x=0.3~3.7（宽 3.4 不变）─
    ax.text(2.0, 10.55, '一、节点对原始特征',
            fontsize=HEADER_FS, fontweight='bold', color='#0B1F4B', ha='center')
    feats = [
        (8.95, 'exp(-|h_src - h_dst|)\n[emb_dim]', '#F4F7FB', '#3A5BAA'),
        (7.65, 'rel_embedding\n[gsl_rel_emb_dim]\n(gsl_has_edge_emb=True)', '#F4F7FB', '#3A5BAA'),
        (6.35, 'is_original_edge\n[1]', '#F4F7FB', '#3A5BAA'),
        (5.00, 'log1p(src_degree)\nlog1p(dst_degree)\n[1+1]', '#F4F7FB', '#3A5BAA'),
        (3.45, 'src_role_emb / dst_role_emb\n[role_emb_dim × 2]\nrole_emb_dim = max(4, hidden_dim/4)', '#EFFAEF', '#2E8B57'),
    ]
    for (y, t, fc, ec) in feats:
        add_box(ax, (0.3, y), 3.4, 1.20, t, fc=fc, ec=ec, fontsize=BOX_FS)

    # ── 2. Concat 张量 ─x=4.2~6.6（宽 2.4 不变）─
    add_box(ax, (4.2, 6.10), 2.4, 2.10,
            'Concat\n→ adaptive_features\n[input_dim]\n\ninput_dim = emb_dim + 3\n+ gsl_rel_emb_dim\n+ 2 × role_emb_dim',
            fc='#FFF7E6', ec='#E08600', fontsize=BOX_FS)

    # 各特征 → Concat
    for y in [9.55, 8.25, 6.95, 5.60, 4.05]:
        arrow(ax, (3.7, y), (4.2, 7.15))

    # ── 3. Encoder 主干 ─x=7.05~10.05（宽 3.0 不变）─
    ax.text(8.55, 10.55, '二、Encoder（共享）',
            fontsize=HEADER_FS, fontweight='bold', color='#0B1F4B', ha='center')

    encoder_blocks = [
        (8.85, 'Linear(input_dim → hidden_dim)\nhidden_dim = MLP_hidden_dim'),
        (7.65, 'LayerNorm(hidden_dim)'),
        (6.55, 'LeakyReLU(0.01)'),
        (5.45, 'Dropout(p=MLP_dropout)'),
        (4.30, 'Linear(hidden_dim → hidden_dim)'),
        (3.20, 'LeakyReLU(0.01)'),
    ]
    for (y, t) in encoder_blocks:
        add_box(ax, (7.05, y), 3.0, 0.95, t,
                fc='#EAF2FB', ec='#1F49D8', fontsize=BOX_FS)

    # Concat → Encoder
    arrow(ax, (6.6, 7.15), (7.05, 9.32), lw=1.8)

    # Encoder 内部串联
    for y_top, y_bot in [(8.85, 8.60), (7.65, 7.50), (6.55, 6.40), (5.45, 5.25), (4.30, 4.15)]:
        arrow(ax, (8.55, y_top), (8.55, y_bot))

    # hidden 输出节点
    add_box(ax, (7.05, 1.95), 3.0, 0.95, 'hidden  [hidden_dim]',
            fc='#EFFAEF', ec='#2E8B57', fontsize=BOX_FS + 1, bold=True)
    arrow(ax, (8.55, 3.20), (8.55, 2.90), lw=1.8)

    # ── 4. 三路输出头 ─x=10.5~12.8（宽 2.3 不变）─
    ax.text(11.65, 10.55, '三、三路输出头',
            fontsize=HEADER_FS, fontweight='bold', color='#0B1F4B', ha='center')

    heads = [
        (8.85, 'gate_head\nLinear\n(hidden_dim → 1)', '#EAF2FB', '#1F49D8', '$s_g$  基础门控'),
        (6.55, 'denoise_head\nLinear\n(hidden_dim → 1)', '#FCEEF1', '#C0392B', '$s_d$  去噪分支'),
        (4.30, 'completion_head\nLinear\n(hidden_dim → 1)', '#FCEEF1', '#C0392B', '$s_c$  补全分支'),
    ]
    for (y, t, fc, ec, label) in heads:
        add_box(ax, (10.5, y), 2.3, 1.30, t, fc=fc, ec=ec, fontsize=BOX_FS - 1)
        ax.text(11.65, y - 0.40, label, ha='center', va='center',
                fontsize=BOX_FS, color='#0B1F4B', fontweight='bold')

    # hidden → 三路 head
    arrow(ax, (10.05, 2.40), (10.5, 9.40), color='#2E8B57', lw=1.6)
    arrow(ax, (10.05, 2.40), (10.5, 7.10), color='#2E8B57', lw=1.6)
    arrow(ax, (10.05, 2.40), (10.5, 4.85), color='#2E8B57', lw=1.6)

    # ── 5. 融合公式 ─宽 6.0 不变 ──────────────────
    add_box(ax, (3.5, 0.55), 6.0, 1.25,
            r'$w_{ij}=\sigma(\,s_g + \alpha_d\cdot \mathbf{1}_{\mathrm{denoise}}\cdot s_d + \alpha_c\cdot \mathbf{1}_{\mathrm{compl}}\cdot s_c\,)$' + '\n通过 use_denoise / use_completion 开关解耦"去噪"与"补全"',
            fc='#FFF7E6', ec='#E08600', fontsize=BOX_FS + 1)

    # 三路 head → 融合公式
    arrow(ax, (11.65, 8.65), (9.5, 1.80), color='#888', lw=1.3)
    arrow(ax, (11.65, 6.35), (9.5, 1.80), color='#888', lw=1.3)
    arrow(ax, (11.65, 4.10), (9.5, 1.80), color='#888', lw=1.3)

    # ── 图例 ──────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor='#F4F7FB', edgecolor='#3A5BAA', label='原 KnowDDI 已有特征'),
        mpatches.Patch(facecolor='#EFFAEF', edgecolor='#2E8B57', label='本文新增字段 / 中间张量'),
        mpatches.Patch(facecolor='#EAF2FB', edgecolor='#1F49D8', label='Encoder / gate_head（保留+解耦）'),
        mpatches.Patch(facecolor='#FCEEF1', edgecolor='#C0392B', label='本文新增 denoise_head / completion_head'),
        mpatches.Patch(facecolor='#FFF7E6', edgecolor='#E08600', label='Concat / 融合公式'),
    ]
    ax.legend(handles=legend_handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, fontsize=LEGEND_FS)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    print(f'[完成] 图4-1 已保存至: {OUT}')


if __name__ == '__main__':
    main()
