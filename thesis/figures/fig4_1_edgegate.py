"""
图4-1 EdgeGateNetwork 内部结构示意图
普通模型框架图：输入特征 → 共享编码器（MLP） → 三路输出头 → 边权融合
不含代码变量名，以神经网络层方框表示各模块
"""  # 修改新增：改为普通模型框架图，去除代码标注

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig4_1_edgegate.png')


def add_box(ax, xy, w, h, text, fc='#EAF2FB', ec='#1F49D8', fontsize=11, bold=False, lw=1.5):
    box = FancyBboxPatch((xy[0], xy[1]), w, h,
                         boxstyle='round,pad=0.03,rounding_size=0.07',
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', color='#0B1F4B',
            multialignment='center')


def arrow(ax, p1, p2, color='#555', lw=1.6, style='->'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style,
                                 mutation_scale=14, linewidth=lw, color=color))


def main():
    fig, ax = plt.subplots(figsize=(13, 10))  # 修改新增：增加图形高度避免底部文字与方框重叠
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.8, 11)  # 修改新增：下限留空给底部公式框和图例
    ax.axis('off')

    TITLE_FS = 15
    HEAD_FS = 13
    BOX_FS = 12
    SMALL_FS = 11

    # ── 标题 ──────────────────────────────────────────
    ax.text(6.5, 10.65, '图4-1  EdgeGateNetwork 结构示意图',
            ha='center', va='center', fontsize=TITLE_FS, fontweight='bold', color='#0B1F4B')

    # ══════════════════════════════════════════════════
    # 第一列：输入特征（三类）
    # ══════════════════════════════════════════════════
    ax.text(1.85, 9.90, '输入特征', ha='center', va='center',
            fontsize=HEAD_FS, fontweight='bold', color='#0B1F4B')

    feat_boxes = [
        (8.50, '药物节点嵌入\n（源节点 / 目标节点）', '#F0F4FC', '#3A5BAA'),
        (7.00, '关系类型嵌入', '#F0F4FC', '#3A5BAA'),
        (5.55, '原始边标志\n（是否为已知边）', '#F0F4FC', '#3A5BAA'),
        (4.10, '节点度数特征\n（源 / 目标）', '#F0F4FC', '#3A5BAA'),
        (2.55, '节点角色嵌入\n（源 / 目标）', '#EFFAEF', '#2E8B57'),
    ]
    for (y, t, fc, ec) in feat_boxes:
        add_box(ax, (0.25, y), 3.2, 1.10, t, fc=fc, ec=ec, fontsize=BOX_FS)

    # ── 特征拼接 ────────────────────────────────────
    add_box(ax, (3.90, 5.50), 2.40, 2.50,
            '特征\n拼接\n(Concat)',
            fc='#FFF7E6', ec='#E08600', fontsize=BOX_FS, bold=True)

    # 各特征 → Concat
    for y_src in [9.05, 7.55, 6.10, 4.65, 3.10]:
        arrow(ax, (3.45, y_src), (3.90, 6.75))

    # ══════════════════════════════════════════════════
    # 第二列：共享编码器（MLP）
    # ══════════════════════════════════════════════════
    ax.text(8.20, 9.90, '共享编码器（MLP）', ha='center', va='center',
            fontsize=HEAD_FS, fontweight='bold', color='#0B1F4B')

    encoder_layers = [
        (8.70, '全连接层\n(Linear)'),
        (7.45, '层归一化\n(LayerNorm)'),
        (6.25, '激活函数\n(LeakyReLU)'),
        (5.05, '随机丢弃\n(Dropout)'),
        (3.85, '全连接层\n(Linear)'),
        (2.70, '激活函数\n(LeakyReLU)'),
    ]
    for (y, t) in encoder_layers:
        add_box(ax, (6.55, y), 3.30, 1.00, t,
                fc='#EAF2FB', ec='#1F49D8', fontsize=BOX_FS)

    # Concat → 第一个编码层
    arrow(ax, (6.30, 6.75), (6.55, 9.20), lw=1.8, color='#555')

    # 编码器内部串联
    for y_top, y_bot in [(8.70, 8.45), (7.45, 7.25), (6.25, 6.05),
                         (5.05, 4.85), (3.85, 3.70)]:
        arrow(ax, (8.20, y_top), (8.20, y_bot))

    # 隐层输出
    add_box(ax, (6.55, 1.60), 3.30, 0.90, '隐层表示  h',
            fc='#EFFAEF', ec='#2E8B57', fontsize=BOX_FS + 1, bold=True)
    arrow(ax, (8.20, 2.70), (8.20, 2.50), lw=1.8)

    # ══════════════════════════════════════════════════
    # 第三列：三路输出头
    # ══════════════════════════════════════════════════
    ax.text(11.50, 9.90, '三路输出头', ha='center', va='center',
            fontsize=HEAD_FS, fontweight='bold', color='#0B1F4B')

    heads = [
        (8.50, '基础\n门控头', '$s_g$', '#EAF2FB', '#1F49D8'),
        (6.20, '去噪\n输出头', '$s_d$', '#FCEEF1', '#C0392B'),
        (3.90, '补全\n输出头', '$s_c$', '#FCEEF1', '#C0392B'),
    ]
    for (y, t, label, fc, ec) in heads:
        add_box(ax, (10.20, y), 2.50, 1.00, t, fc=fc, ec=ec, fontsize=BOX_FS)
        ax.text(11.45, y - 0.35, label, ha='center', va='center',
                fontsize=BOX_FS + 1, color=ec, fontweight='bold')

    # h → 三路 head
    for y_dst in [9.00, 6.70, 4.40]:
        arrow(ax, (9.85, 2.05), (10.20, y_dst), color='#2E8B57', lw=1.5)

    # ══════════════════════════════════════════════════
    # 底部融合公式
    # ══════════════════════════════════════════════════
    add_box(ax, (3.20, -0.50), 6.60, 1.20,  # 修改新增：下移融合公式框，避免与箭头重叠
            r'$w_{ij}=\sigma\!\left(s_g+\alpha_d\cdot\mathbf{1}_{\mathrm{den}}\cdot s_d' +
            r'+\alpha_c\cdot\mathbf{1}_{\mathrm{comp}}\cdot s_c\right)$' + '\n边权融合（指示变量控制去噪 / 补全是否生效）',
            fc='#FFF7E6', ec='#E08600', fontsize=BOX_FS)

    # 三路 head → 融合公式
    for y_src in [9.00, 6.70, 4.40]:
        arrow(ax, (11.45, y_src), (9.80, 0.10), color='#888', lw=1.2)  # 修改新增：箭头终点也下移配合

    # ── 图例 ──────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor='#F0F4FC', edgecolor='#3A5BAA', label='原有药物/关系特征'),
        mpatches.Patch(facecolor='#EFFAEF', edgecolor='#2E8B57', label='本文新增角色嵌入 / 隐层输出'),
        mpatches.Patch(facecolor='#EAF2FB', edgecolor='#1F49D8', label='共享编码器 / 基础门控头'),
        mpatches.Patch(facecolor='#FCEEF1', edgecolor='#C0392B', label='本文新增去噪 / 补全输出头'),
        mpatches.Patch(facecolor='#FFF7E6', edgecolor='#E08600', label='特征拼接 / 边权融合'),
    ]
    ax.legend(handles=legend_handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=SMALL_FS)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    print(f'[完成] 图4-1 已保存至: {OUT}')


if __name__ == '__main__':
    main()

