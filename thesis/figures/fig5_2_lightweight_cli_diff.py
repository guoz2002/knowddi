# -*- coding: utf-8 -*-
"""图5-2 三类轻量化方案架构对比示意图（普通模型框架图）
   拓扑稀疏采样 / 特征维度压缩 / 协同优化 三条路径的数据流与压缩点
   不含命令行参数，以神经网络模块方框展示各压缩入口
"""  # 修改新增：改为普通模型框架图，去除命令行参数表格
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC',
                                   'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False



def add_box(ax, xy, w, h, text, fc, ec, fs=12, fw='normal', tc='#0B1F4B', lw=1.5):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.03,rounding_size=0.07",
                         facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight=fw, multialignment='center')


def add_arrow(ax, p1, p2, color='#555', lw=1.5, style='-|>', ls='-'):
    arr = FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                          linewidth=lw, linestyle=ls, mutation_scale=13)
    ax.add_patch(arr)


def main():
    BOX_FS = 11
    HEAD_FS = 13
    TITLE_FS = 15

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # ── 标题 ────────────────────────────────────────
    ax.text(7.5, 10.65, '图5-2  三类轻量化方案的模型架构对比示意',
            ha='center', va='center', fontsize=TITLE_FS,
            fontweight='bold', color='#0B1F4B')

    # ══════════════════════════════════════════════
    # 顶部：共享输入层（BKG 子图）
    # ══════════════════════════════════════════════
    add_box(ax, (5.25, 9.50), 4.50, 0.80,
            'BKG 子图提取（以药物对为中心的局部子图）',
            fc='#F4F7FB', ec='#7A8AA8', fs=BOX_FS + 1, fw='bold')

    # ══════════════════════════════════════════════
    # 三列：基线 / 拓扑稀疏 / 特征压缩 / 协同优化
    # ══════════════════════════════════════════════
    col_info = [
        # x0, title, topo_compress, dim_compress, fc, ec
        (0.30,  '基线',      False, False, '#F4F7FB', '#7A8AA8'),
        (4.05,  'A 拓扑稀疏采样', True,  False, '#FCEEF1', '#C0392B'),
        (7.80,  'B 特征维度压缩', False, True,  '#EAF2FB', '#1F49D8'),
        (11.55, 'C 协同优化',  True,  True,  '#EFFAEF', '#2E8B57'),
    ]

    for (x0, title, topo, dim, fc, ec) in col_info:
        W = 3.20
        BX = x0 + 0.05

        # 列标题
        add_box(ax, (x0, 8.80), W, 0.55, title,
                fc=fc, ec=ec, fs=HEAD_FS, fw='bold', lw=2.0)

        # 箭头：子图提取 → 拓扑采样
        add_arrow(ax, (7.5, 9.50), (x0 + W / 2, 9.35), color='#999', lw=1.2)

        # 拓扑采样模块
        if topo:
            t_fc, t_ec, t_fw = '#FCEEF1', '#C0392B', 'bold'
            t_txt = '子图采样\n（稀疏采样，缩小规模）'
        else:
            t_fc, t_ec, t_fw = '#FFFFFF', '#AAAAAA', 'normal'
            t_txt = '子图采样\n（标准采样）'
        add_box(ax, (BX, 7.65), W - 0.10, 0.90,
                t_txt, fc=t_fc, ec=t_ec, fs=BOX_FS, fw=t_fw)
        add_arrow(ax, (x0 + W / 2, 7.65), (x0 + W / 2, 7.40))

        # GraphSAGE 全局编码
        add_box(ax, (BX, 6.55), W - 0.10, 0.75,
                'GraphSAGE\n全局节点编码',
                fc='#EAF2FB', ec='#1F49D8', fs=BOX_FS)
        add_arrow(ax, (x0 + W / 2, 6.55), (x0 + W / 2, 6.30))

        # 特征维度模块
        if dim:
            d_fc, d_ec, d_fw = '#EAF2FB', '#1F49D8', 'bold'
            d_txt = '节点嵌入\n（压缩维度）'
        else:
            d_fc, d_ec, d_fw = '#FFFFFF', '#AAAAAA', 'normal'
            d_txt = '节点嵌入\n（标准维度）'
        add_box(ax, (BX, 5.45), W - 0.10, 0.75,
                d_txt, fc=d_fc, ec=d_ec, fs=BOX_FS, fw=d_fw)
        add_arrow(ax, (x0 + W / 2, 5.45), (x0 + W / 2, 5.20))

        # GSL 图结构学习
        if dim:
            g_fc, g_ec = '#EAF2FB', '#1F49D8'
            g_txt = '图结构学习（GSL）\n（压缩隐层维度）'
        else:
            g_fc, g_ec = '#FFFFFF', '#AAAAAA'
            g_txt = '图结构学习（GSL）\n（标准隐层维度）'
        add_box(ax, (BX, 4.35), W - 0.10, 0.75,
                g_txt, fc=g_fc, ec=g_ec, fs=BOX_FS)
        add_arrow(ax, (x0 + W / 2, 4.35), (x0 + W / 2, 4.10))

        # 分类器 MLP
        if dim:
            m_fc, m_ec, m_fw = '#EAF2FB', '#1F49D8', 'bold'
            m_txt = '分类器 MLP\n（压缩隐层）'
        else:
            m_fc, m_ec, m_fw = '#FFFFFF', '#AAAAAA', 'normal'
            m_txt = '分类器 MLP\n（标准隐层）'
        add_box(ax, (BX, 3.25), W - 0.10, 0.75,
                m_txt, fc=m_fc, ec=m_ec, fs=BOX_FS, fw=m_fw)
        add_arrow(ax, (x0 + W / 2, 3.25), (x0 + W / 2, 3.00))

        # 输出层
        add_box(ax, (BX, 2.15), W - 0.10, 0.75,
                'DDI 关系预测\n（Softmax 输出）',
                fc='#EFFAEF', ec='#2E8B57', fs=BOX_FS)

        # 压缩点标注
        points = []
        if topo:
            points.append('↑ 拓扑压缩入口')
        if dim:
            points.append('↑ 维度压缩入口')
        if points:
            ax.text(x0 + W / 2, 1.55, '  /  '.join(points),
                    ha='center', va='center', fontsize=9.5,
                    color=ec, fontweight='bold')

        # 列标注摘要
        summary_parts = []
        if topo:
            summary_parts.append('子图规模 ↓')
        if dim:
            summary_parts.append('特征维度 ↓')
        if not summary_parts:
            summary_parts = ['原始规模']
        ax.text(x0 + W / 2, 1.10,
                ' | '.join(summary_parts),
                ha='center', va='center', fontsize=10,
                color='#5A6B8C', style='italic')

    # ── 列间分隔线 ──────────────────────────────────
    for x in [3.90, 7.65, 11.40]:
        ax.plot([x, x], [0.80, 9.45], '--', color='#CCCCCC', lw=0.9, zorder=0)

    # ── 底部说明 ────────────────────────────────────
    ax.text(7.5, 0.45,
            '高亮模块为相对基线发生变化的压缩点；基线沿用 KnowDDI 原始配置',
            ha='center', va='center', fontsize=10,
            color='#5A6B8C', style='italic')

    # ── 图例 ────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor='#FFFFFF', edgecolor='#AAAAAA',
                       label='与基线一致（未压缩）'),
        mpatches.Patch(facecolor='#FCEEF1', edgecolor='#C0392B',
                       label='拓扑压缩入口（子图稀疏采样）'),
        mpatches.Patch(facecolor='#EAF2FB', edgecolor='#1F49D8',
                       label='维度压缩入口（嵌入 / 隐层缩减）'),
        mpatches.Patch(facecolor='#EFFAEF', edgecolor='#2E8B57',
                       label='关系预测输出'),
    ]
    ax.legend(handles=handles, loc='lower center', fontsize=BOX_FS,
              ncol=4, frameon=True, framealpha=0.92,
              edgecolor='#B7C0D4', bbox_to_anchor=(0.5, -0.03))

    out_path = os.path.join(os.path.dirname(__file__),
                            'fig5_2_lightweight_cli_diff.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[完成] 图5-2 已保存至: ' + out_path)


if __name__ == '__main__':
    main()

