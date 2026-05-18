# -*- coding: utf-8 -*-
"""图5-2 三类轻量化方案的命令行参数差异表与改动路径示意。
   —— B：拓扑稀疏 ；C：维度压缩 ；D：联合压缩 ——"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC',
                                   'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def add_box(ax, xy, w, h, text, fc, ec, fs=12, fw='normal', tc='#0B1F4B'):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                         facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight=fw)


def add_arrow(ax, p1, p2, color='#0B1F4B', lw=1.6, style='-|>', ls='-'):
    arr = FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                          linewidth=lw, linestyle=ls,
                          mutation_scale=14)
    ax.add_patch(arr)


def main():
    BOX_FS = 13
    HEADER_FS = 14
    TITLE_FS = 16

    fig, ax = plt.subplots(figsize=(15, 9.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.3, 11.5)
    ax.axis('off')

    # —— 上半区：参数差异表 ——
    table_x0 = 0.5
    col_widths = [4.0, 2.6, 2.6, 2.6, 2.6]   # 参数名 / A / B / C / D
    col_xs = [table_x0]
    for w in col_widths[:-1]:
        col_xs.append(col_xs[-1] + w)

    # 表头
    headers = ['命令行参数', '基线', 'A 拓扑稀疏采样', 'B 参数压缩', 'C 协同优化']
    header_colors = ['#F4F7FB', '#F4F7FB', '#FCEEF1', '#EAF2FB', '#EFFAEF']
    header_edges  = ['#3A5BAA', '#3A5BAA', '#C0392B', '#1F49D8', '#2E8B57']
    header_y = 9.5
    header_h = 0.9
    for x, w, t, fc, ec in zip(col_xs, col_widths, headers,
                               header_colors, header_edges):
        add_box(ax, (x, header_y), w, header_h, t,
                fc=fc, ec=ec, fs=HEADER_FS, fw='bold')

    # 数据行（基线 / A / B / C 取值，与 pytorch/train.py 默认及第四章复现设置保持一致）
    rows = [
        ('--max_nodes_per_hop', '200',    '100',    '200',    '100',    'topo'),
        ('--max_links',         '250000', '50000',  '250000', '50000',  'topo'),
        ('--emb_dim',           '32',     '32',     '16',     '16',     'dim'),
        ('--gsl_rel_emb_dim',   '8',      '8',      '4',      '4',      'dim'),
        ('--MLP_hidden_dim',    '64',     '64',     '32',     '32',     'dim'),
    ]
    row_h = 0.85
    row_y0 = header_y - 0.15
    for i, (name, a, b, c, d, kind) in enumerate(rows):
        y = row_y0 - (i + 1) * row_h
        # 参数名格
        add_box(ax, (col_xs[0], y), col_widths[0], row_h, name,
                fc='#FFFFFF', ec='#7A8AA8', fs=BOX_FS,
                fw='bold', tc='#0B1F4B')
        # A 基线
        add_box(ax, (col_xs[1], y), col_widths[1], row_h, a,
                fc='#FFFFFF', ec='#7A8AA8', fs=BOX_FS, tc='#5A6B8C')
        # A / B / C —— 改动了就高亮
        for k, val, base in zip([2, 3, 4], [b, c, d], [a, a, a]):
            changed = (val != base)
            fc = ['#FCEEF1', '#EAF2FB', '#EFFAEF'][k - 2] if changed else '#FFFFFF'
            ec = ['#C0392B', '#1F49D8', '#2E8B57'][k - 2] if changed else '#7A8AA8'
            tc = ['#7C2C20', '#16306E', '#1F5E3D'][k - 2] if changed else '#5A6B8C'
            fw = 'bold' if changed else 'normal'
            add_box(ax, (col_xs[k], y), col_widths[k], row_h, val,
                    fc=fc, ec=ec, fs=BOX_FS, fw=fw, tc=tc)

    # 行分组：拓扑参数 / 维度参数 侧标
    ax.text(col_xs[0] - 0.2, row_y0 - 1 * row_h - row_h * 0.5,
            '拓\u3000扑\n参\u3000数', ha='right', va='center',
            fontsize=BOX_FS, color='#C0392B', fontweight='bold')
    ax.text(col_xs[0] - 0.2, row_y0 - 4 * row_h - row_h * 0.5,
            '维\u3000度\n参\u3000数', ha='right', va='center',
            fontsize=BOX_FS, color='#1F49D8', fontweight='bold')

    # —— 下半区：改动路径示意（三个 mini 流程） ——
    panel_y = 0.6
    panel_h = 3.2
    panel_titles = ['A 拓扑稀疏采样：只动子图采样',
                    'B 参数压缩：只动特征维度',
                    'C 协同优化：双入口同时收紧']
    panel_edges = ['#C0392B', '#1F49D8', '#2E8B57']
    panel_fcs   = ['#FCEEF1', '#EAF2FB', '#EFFAEF']

    panel_xs = [0.5, 5.7, 10.9]
    panel_w = 4.6

    for px, title, ec, fc in zip(panel_xs, panel_titles,
                                 panel_edges, panel_fcs):
        # 外框
        outer = FancyBboxPatch((px, panel_y), panel_w, panel_h,
                               boxstyle="round,pad=0.04,rounding_size=0.10",
                               facecolor=fc, edgecolor=ec,
                               linewidth=1.8, alpha=0.55)
        ax.add_patch(outer)
        ax.text(px + panel_w / 2, panel_y + panel_h - 0.35, title,
                ha='center', va='center', fontsize=HEADER_FS,
                fontweight='bold', color=ec)

    # A 路径：BFS 采样 → max_nodes_per_hop & max_links 截断 → 学习子图 (维度链不变)
    bx = panel_xs[0]
    add_box(ax, (bx + 0.25, panel_y + 1.45), panel_w - 0.5, 0.85,
            'BFS 子图采样\n'
            'max_nodes_per_hop 100 / max_links 50000',
            fc='#FCEEF1', ec='#C0392B', fs=BOX_FS - 1, fw='bold', tc='#7C2C20')
    add_box(ax, (bx + 0.25, panel_y + 0.40), panel_w - 0.5, 0.75,
            '维度链保持 32 / 8 / 64 不变',
            fc='#FFFFFF', ec='#7A8AA8', fs=BOX_FS - 1, tc='#5A6B8C')
    add_arrow(ax, (bx + panel_w / 2, panel_y + 1.45),
              (bx + panel_w / 2, panel_y + 1.15), color='#7A8AA8', lw=1.4)

    # B 路径：BFS 采样不变 → 维度链 16 / 4 / 32
    cx = panel_xs[1]
    add_box(ax, (cx + 0.25, panel_y + 1.45), panel_w - 0.5, 0.85,
            'BFS 子图采样\n保持 200 / 250000 不变',
            fc='#FFFFFF', ec='#7A8AA8', fs=BOX_FS - 1, tc='#5A6B8C')
    add_box(ax, (cx + 0.25, panel_y + 0.40), panel_w - 0.5, 0.75,
            'emb_dim 16  /  gsl_rel_emb_dim 4  /  MLP_hidden_dim 32',
            fc='#EAF2FB', ec='#1F49D8', fs=BOX_FS - 2, fw='bold', tc='#16306E')
    add_arrow(ax, (cx + panel_w / 2, panel_y + 1.45),
              (cx + panel_w / 2, panel_y + 1.15), color='#7A8AA8', lw=1.4)

    # C 路径：两条都收紧
    dx = panel_xs[2]
    add_box(ax, (dx + 0.25, panel_y + 1.45), panel_w - 0.5, 0.85,
            'BFS 子图采样\nmax_nodes_per_hop 100 / max_links 50000',
            fc='#FCEEF1', ec='#C0392B', fs=BOX_FS - 1, fw='bold', tc='#7C2C20')
    add_box(ax, (dx + 0.25, panel_y + 0.40), panel_w - 0.5, 0.75,
            '维度链：16 / 4 / 32',
            fc='#EAF2FB', ec='#1F49D8', fs=BOX_FS - 1, fw='bold', tc='#16306E')
    add_arrow(ax, (dx + panel_w / 2, panel_y + 1.45),
              (dx + panel_w / 2, panel_y + 1.15), color='#2E8B57', lw=1.6)

    # 三方案侧的改动入口数标注（放在外框下方，避免与内部 box 重叠）
    for px, k_topo, k_dim in zip(panel_xs, [2, 0, 2], [0, 3, 3]):
        ax.text(px + panel_w / 2, panel_y - 0.15,
                f'拓扑改动 {k_topo} 项  ｜  维度改动 {k_dim} 项',
                ha='center', va='top', fontsize=BOX_FS - 1,
                color='#0B1F4B', fontweight='bold')

    # —— 右侧汇总：改动入口数饼图替代为 mini 文字提示 ——
    add_box(ax, (15.6, panel_y), 0, 0, '', fc='none', ec='none')

    # 顶部标题 / 底部说明
    ax.text(8, 11.05, '图5-2  三类轻量化方案的命令行参数差异表与改动路径示意',
            ha='center', va='center', fontsize=TITLE_FS,
            fontweight='bold', color='#0B1F4B')
    ax.text(8, 4.1,
            '注：“基线”沿用原 KnowDDI 设置；A / B / C 三组方案分别从'
            '“拓扑入口”“参数入口”“双入口”三个角度收紧模型规模，'
            '高亮单元格即为相对基线的改动项。',
            ha='center', va='center', fontsize=BOX_FS - 1,
            color='#5A6B8C', style='italic')

    # 图例
    handles = [
        mpatches.Patch(facecolor='#FFFFFF', edgecolor='#7A8AA8',
                       label='与基线一致（未改动）'),
        mpatches.Patch(facecolor='#FCEEF1', edgecolor='#C0392B',
                       label='拓扑入口改动（A / C）'),
        mpatches.Patch(facecolor='#EAF2FB', edgecolor='#1F49D8',
                       label='参数入口改动（B / C）'),
        mpatches.Patch(facecolor='#EFFAEF', edgecolor='#2E8B57',
                       label='双入口同时改动（C）'),
    ]
    ax.legend(handles=handles, loc='lower center', fontsize=BOX_FS,
              ncol=4, frameon=True, framealpha=0.92,
              edgecolor='#B7C0D4', bbox_to_anchor=(0.5, -0.02))

    out_path = os.path.join(os.path.dirname(__file__),
                            'fig5_2_lightweight_cli_diff.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[完成] 图5-2 已保存至: {out_path}')


if __name__ == '__main__':
    main()
