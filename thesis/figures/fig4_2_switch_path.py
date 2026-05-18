"""
图4-2 四组消融变体的图结构学习流程对比
普通模型框架图：展示 A/B/C/D 四组变体中去噪模块和补全模块的激活状态
不含命令行参数，以模块激活/禁用的方框图形式展示
"""  # 修改新增：改为普通模型框架图，去除CLI参数标注

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig4_2_switch_path.png')


def add_box(ax, xy, w, h, text, fc='#EAF2FB', ec='#1F49D8', fontsize=11,
            bold=False, lw=1.5, alpha=1.0, tc='#0B1F4B'):
    box = FancyBboxPatch((xy[0], xy[1]), w, h,
                         boxstyle='round,pad=0.03,rounding_size=0.07',
                         linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', color=tc,
            multialignment='center', alpha=alpha)


def arrow(ax, p1, p2, color='#555', lw=1.5, style='->', alpha=1.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style,
                                 mutation_scale=13, linewidth=lw,
                                 color=color, alpha=alpha))


def draw_variant(ax, x0, variant_label, variant_name, use_denoise, use_completion,
                 var_fc, var_ec):
    W = 2.80
    BOX_W = W - 0.10
    BX = x0 + 0.05
    BOX_FS = 10.5
    LABEL_FS = 11

    add_box(ax, (x0, 9.60), W, 0.80,
            f'{variant_name}',  # 修改新增：去除 A/B/C/D 字母标签，只保留中文名称
            fc=var_fc, ec=var_ec, fontsize=LABEL_FS, bold=True, lw=2.0)

    add_box(ax, (BX, 8.55), BOX_W, 0.75,
            '局部子图\n（药物对 + BKG邻居）',
            fc='#F4F7FB', ec='#7A8AA8', fontsize=BOX_FS)
    arrow(ax, (x0 + W / 2, 8.55), (x0 + W / 2, 8.20))

    add_box(ax, (BX, 7.35), BOX_W, 0.75,
            'GraphSAGE\n节点编码',
            fc='#EAF2FB', ec='#1F49D8', fontsize=BOX_FS)
    arrow(ax, (x0 + W / 2, 7.35), (x0 + W / 2, 7.00))

    add_box(ax, (BX, 6.15), BOX_W, 0.75,
            '基础门控打分\n$s_g$',
            fc='#EAF2FB', ec='#1F49D8', fontsize=BOX_FS)
    arrow(ax, (x0 + W / 2, 6.15), (x0 + W / 2, 5.80))

    if use_denoise:
        d_fc, d_ec, d_alpha, d_tc = '#FCEEF1', '#C0392B', 1.0, '#0B1F4B'
        d_text = '去噪打分\n$s_d$（激活）'
    else:
        d_fc, d_ec, d_alpha, d_tc = '#F0F0F0', '#AAAAAA', 0.55, '#888888'
        d_text = '去噪打分\n$s_d$（禁用）'
    add_box(ax, (BX, 4.95), BOX_W, 0.75,
            d_text, fc=d_fc, ec=d_ec, fontsize=BOX_FS, alpha=d_alpha, tc=d_tc)
    arrow(ax, (x0 + W / 2, 4.95), (x0 + W / 2, 4.60), alpha=0.9 if use_denoise else 0.3)

    if use_completion:
        c_fc, c_ec, c_alpha, c_tc = '#FCEEF1', '#C0392B', 1.0, '#0B1F4B'
        c_text = '补全打分\n$s_c$（激活）'
    else:
        c_fc, c_ec, c_alpha, c_tc = '#F0F0F0', '#AAAAAA', 0.55, '#888888'
        c_text = '补全打分\n$s_c$（禁用）'
    add_box(ax, (BX, 3.75), BOX_W, 0.75,
            c_text, fc=c_fc, ec=c_ec, fontsize=BOX_FS, alpha=c_alpha, tc=c_tc)
    arrow(ax, (x0 + W / 2, 3.75), (x0 + W / 2, 3.40), alpha=0.9 if use_completion else 0.3)

    add_box(ax, (BX, 2.55), BOX_W, 0.75,
            '边权融合\n$w_{ij} = \\sigma(\\cdot)$',
            fc='#FFF7E6', ec='#E08600', fontsize=BOX_FS)
    arrow(ax, (x0 + W / 2, 2.55), (x0 + W / 2, 2.20))

    add_box(ax, (BX, 1.35), BOX_W, 0.75,
            '自适应图结构\n→ 关系预测',
            fc='#EFFAEF', ec='#2E8B57', fontsize=BOX_FS)

    alpha_d_str = r'$\alpha_d s_d$' if use_denoise else r'$0$'
    alpha_c_str = r'$\alpha_c s_c$' if use_completion else r'$0$'
    eq = r'$s_g + $' + alpha_d_str + r'$ + $' + alpha_c_str
    ax.text(x0 + W / 2, 0.85, eq,
            ha='center', va='center', fontsize=9.5, color='#5A3000')


def main():
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 11.5)
    ax.axis('off')

    ax.text(6.5, 11.15, '图4-2  四组消融变体的图结构学习流程对比',
            ha='center', va='center', fontsize=15,
            fontweight='bold', color='#0B1F4B')

    variants = [
        (0.30, '基线',      False, False, '#F4F7FB', '#3A5BAA'),
        (3.40, '仅去噪',    True,  False, '#FCEEF1', '#C0392B'),
        (6.50, '仅补全',    False, True,  '#EAF2FB', '#1F49D8'),
        (9.60, '去噪+补全', True,  True,  '#EFFAEF', '#2E8B57'),
    ]
    for (x0, name, ud, uc, fc, ec) in variants:
        draw_variant(ax, x0, '', name, ud, uc, fc, ec)  # 修改新增：label 传空字符串，只显示中文名

    for x in [3.30, 6.40, 9.50]:
        ax.plot([x, x], [0.60, 10.50], '--', color='#CCCCCC', lw=0.9, zorder=0)

    ax.text(6.5, 0.30,
            '四组变体共享同一 EdgeGateNetwork 参数，仅通过指示变量控制去噪/补全项是否参与边权融合',
            ha='center', va='center', fontsize=10, color='#5A6B8C', style='italic')

    legend_handles = [
        mpatches.Patch(facecolor='#EAF2FB', edgecolor='#1F49D8', label='GraphSAGE编码 / 基础门控（共享）'),
        mpatches.Patch(facecolor='#FCEEF1', edgecolor='#C0392B', label='去噪/补全打分（激活）'),
        mpatches.Patch(facecolor='#F0F0F0', edgecolor='#AAAAAA', label='去噪/补全打分（禁用）'),
        mpatches.Patch(facecolor='#FFF7E6', edgecolor='#E08600', label='边权融合'),
        mpatches.Patch(facecolor='#EFFAEF', edgecolor='#2E8B57', label='自适应图结构 → 关系预测'),
    ]
    ax.legend(handles=legend_handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    print(f'[完成] 图4-2 已保存至: {OUT}')


if __name__ == '__main__':
    main()
