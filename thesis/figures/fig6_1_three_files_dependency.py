# -*- coding: utf-8 -*-
"""图6-1  本文改动收敛于三个核心文件的依赖关系图。
展示 train.py（CLI 入口）→ subgraph_extraction.py（拓扑稀疏） 与 → gsl_model.EdgeGateNetwork（参数压缩）
之间的数据流向，以及命令行参数如何分别注入这两条链路。
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC',
                               'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


# ---------- 颜色 ----------
C_BLUE_BG, C_BLUE = '#EAF2FB', '#1F49D8'
C_RED_BG,  C_RED  = '#FCEEF1', '#C0392B'
C_GREEN_BG, C_GREEN = '#EFFAEF', '#2E8B57'
C_ORANGE_BG, C_ORANGE = '#FFF7E6', '#E08600'
C_GRAY_BG = '#F4F7FB'
C_TEXT_BLUE = '#16306E'
C_TEXT_RED = '#7C2C20'
C_TEXT_GREEN = '#1F5E3D'
C_TEXT_ORANGE = '#7A4F00'


def draw_box(ax, xy, w, h, label, fc, ec, tc, fs=12, fw='normal', lh=1.0,
             radius=0.18):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f'round,pad=0.02,rounding_size={radius}',
                         linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
            fontsize=fs, color=tc, fontweight=fw, linespacing=lh)


def arrow(ax, p0, p1, color, lw=1.6, style='->', rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style,
                                 color=color, linewidth=lw, linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}',
                                 mutation_scale=18))


def main():
    BOX_FS = 17
    HEADER_FS = 19
    TITLE_FS = 21

    fig, ax = plt.subplots(figsize=(16, 9.5))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # ===== 顶部 CLI 入口 =====
    draw_box(ax, (5.5, 11.6), 5.0, 1.8,
             '入口  train.py\n（命令行参数 argparse\n  + Trainer 主循环）',
             C_BLUE_BG, C_BLUE, C_TEXT_BLUE, fs=HEADER_FS, fw='bold', lh=1.45)

    # CLI 参数列（左上挂在 train.py 下方）
    args_left = [
        '--max_nodes_per_hop',
        '--max_links',
    ]
    args_right = [
        '--emb_dim',
        '--gsl_rel_emb_dim',
        '--MLP_hidden_dim',
    ]
    draw_box(ax, (0.6, 9.0), 5.0, 1.9,
             '拓扑稀疏参数\n' + '\n'.join(args_left),
             C_RED_BG, C_RED, C_TEXT_RED, fs=BOX_FS, fw='bold', lh=1.55)
    draw_box(ax, (10.4, 9.0), 5.6, 1.9,
             '参数压缩参数\n' + '\n'.join(args_right),
             C_GREEN_BG, C_GREEN, C_TEXT_GREEN, fs=BOX_FS, fw='bold', lh=1.55)

    # train.py → 两组参数
    arrow(ax, (7.2, 11.6), (3.1, 10.9), C_BLUE, lw=1.6, rad=-0.18)
    arrow(ax, (8.8, 11.6), (13.2, 10.9), C_BLUE, lw=1.6, rad=0.18)

    # ===== 左路：拓扑稀疏 =====
    draw_box(ax, (0.6, 5.6), 5.0, 2.6,
             '①  data_processor/\nsubgraph_extraction.py\n\n'
             'BFS k-hop\n+ max_nodes_per_hop 截断\n+ max_links 边采样\n→ 学习子图',
             C_RED_BG, C_RED, C_TEXT_RED, fs=BOX_FS, fw='bold', lh=1.5)

    arrow(ax, (3.1, 9.0), (3.1, 8.2), C_RED, lw=1.8)

    draw_box(ax, (0.6, 3.4), 5.0, 1.6,
             '稀疏子图  (sparse_subgraph)\n节点数 ↓   边数 ↓',
             C_GRAY_BG, '#888888', '#333333', fs=BOX_FS, lh=1.5)

    arrow(ax, (3.1, 5.6), (3.1, 5.0), '#666666', lw=1.6)

    # ===== 右路：参数压缩 =====
    draw_box(ax, (10.4, 5.6), 5.6, 2.6,
             '②  model/gsl_model.py\n→ EdgeGateNetwork\n\n'
             'Linear(emb_dim\n        → gsl_rel_emb_dim)\n+ Sigmoid Edge Gate',
             C_GREEN_BG, C_GREEN, C_TEXT_GREEN, fs=BOX_FS, fw='bold', lh=1.5)

    arrow(ax, (13.2, 9.0), (13.2, 8.2), C_GREEN, lw=1.8)

    draw_box(ax, (10.4, 3.4), 5.6, 1.6,
             '压缩边表征  e_uv\n维度  emb_dim → gsl_rel_emb_dim',
             C_GRAY_BG, '#888888', '#333333', fs=BOX_FS, lh=1.5)

    arrow(ax, (13.2, 5.6), (13.2, 5.0), '#666666', lw=1.6)

    # ===== 中央汇合：GraphSAGE → GSL 卷积 → Classifier =====
    draw_box(ax, (5.9, 2.8), 4.4, 2.6,
             'GraphSAGE\n+ gsl_model.GSLModel\n→ Classifier_model.Classifier\n\n'
             '多层消息传递\n+ EdgeGate 加权\n→ DDI logits',
             C_BLUE_BG, C_BLUE, C_TEXT_BLUE, fs=BOX_FS - 1, fw='bold', lh=1.45)

    arrow(ax, (5.6, 3.7), (5.85, 3.7), C_RED, lw=1.8, rad=-0.15)
    arrow(ax, (10.4, 3.7), (10.3, 3.7), C_GREEN, lw=1.8, rad=0.15)

    # ===== 底部输出 + 反馈到 Trainer =====
    draw_box(ax, (5.4, 0.4), 5.4, 1.9,
             '推理输出\n→ inference_total_latency / max_memory_allocated\n'
             '→ manager/trainer.py 累加并写入日志',
             C_ORANGE_BG, C_ORANGE, C_TEXT_ORANGE, fs=BOX_FS, fw='bold', lh=1.5)

    arrow(ax, (8.1, 2.8), (8.1, 2.3), C_ORANGE, lw=1.8)

    # 反馈虚线：trainer → train.py（loop） —— 走最右侧避免与 EdgeGateNetwork 重叠
    arrow(ax, (10.8, 1.5), (16.4, 1.5), '#888888', lw=1.2, ls='--')
    arrow(ax, (16.4, 1.5), (16.4, 12.5), '#888888', lw=1.2, ls='--', rad=0.0)
    arrow(ax, (16.4, 12.5), (10.5, 12.5), '#888888', lw=1.2, ls='--')
    ax.text(16.6, 7.0, '指标回灌\n（每 epoch\n  评估 / 早停）',
            fontsize=BOX_FS - 1, color='#555555', va='center', ha='left',
            linespacing=1.5)

    # ===== 标注：三文件改动汇聚标签（放在中间空白处）
    ax.text(8.1, 7.0, '本文 3 处核心改动\n汇聚于此 3 个文件',
            ha='center', va='center', fontsize=HEADER_FS,
            color='#16306E', fontweight='bold', linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFFFFF',
                      edgecolor='#1F49D8', lw=1.4))

    # 文件标签角标（③ 已显示在顶部 train.py 框上方；① ② 由各自分支的方框内文本承担）
    ax.text(8.0, 13.6, '③  pytorch/train.py · argparse 入口',
            ha='center', va='center', fontsize=BOX_FS - 1,
            color=C_TEXT_BLUE, fontweight='bold')

    # 图例
    legend_handles = [
        mpatches.Patch(facecolor=C_BLUE_BG, edgecolor=C_BLUE,
                       label='CLI / 主流程（train.py · 分类器）'),
        mpatches.Patch(facecolor=C_RED_BG, edgecolor=C_RED,
                       label='拓扑稀疏链路（subgraph_extraction.py）'),
        mpatches.Patch(facecolor=C_GREEN_BG, edgecolor=C_GREEN,
                       label='参数压缩链路（EdgeGateNetwork）'),
        mpatches.Patch(facecolor=C_ORANGE_BG, edgecolor=C_ORANGE,
                       label='输出指标（latency / memory）'),
        mpatches.Patch(facecolor='#FFFFFF', edgecolor='#888888',
                       label='指标回灌（虚线）'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.005), ncol=3, fontsize=BOX_FS,
               frameon=True, edgecolor='#888888')

    fig.suptitle('图6-1  改动收敛于三个核心文件的依赖关系：'
                 'train.py  <->  subgraph_extraction.py  <->  EdgeGateNetwork',
                 fontsize=TITLE_FS, fontweight='bold', y=0.985)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.07)

    out = os.path.join(os.path.dirname(__file__),
                       'fig6_1_three_files_dependency.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[完成] 图6-1 已保存至: {out}')


if __name__ == '__main__':
    main()
