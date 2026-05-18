# -*- coding: utf-8 -*-
"""图5-1 拓扑/参数双入口压缩关系图。
   —— 左：max_nodes_per_hop / max_links 在 BFS 扩展与候选完全图上的截断位置 ——
   —— 右：emb_dim / gsl_rel_emb_dim / MLP_hidden_dim 三者的链式依赖图 ——"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
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


def draw_left(ax):
    """左：拓扑入口"""
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.4, 11)
    ax.axis('off')
    ax.set_title('（a）拓扑入口：子图采样阶段的两处截断',
                 fontsize=15, fontweight='bold', color='#0B1F4B', pad=8)

    HEAD_FS = 13
    BOX_FS = 12

    # 阶段 1：中心药物对
    add_box(ax, (3.4, 9.4), 3.2, 1.0, '中心药物对  (u, v)',
            fc='#F4F7FB', ec='#3A5BAA', fs=BOX_FS, fw='bold')

    # 阶段 2：BFS 1-hop 邻居
    add_box(ax, (1.0, 7.5), 8.0, 1.2,
            'BFS 第 1 跳邻居全集  N_1(u) ∪ N_1(v)',
            fc='#F4F7FB', ec='#3A5BAA', fs=BOX_FS)

    # 截断 1：max_nodes_per_hop
    add_box(ax, (1.0, 6.0), 8.0, 1.0,
            '【截断 ①】 max_nodes_per_hop  →  每跳保留前 K 个邻居',
            fc='#FCEEF1', ec='#C0392B', fs=BOX_FS, fw='bold', tc='#7C2C20')

    # 阶段 3：BFS 2-hop
    add_box(ax, (1.0, 4.5), 8.0, 1.0,
            'BFS 第 2 跳扩展  →  最终候选节点集 V_sub',
            fc='#F4F7FB', ec='#3A5BAA', fs=BOX_FS)

    # 阶段 4：候选完全图
    add_box(ax, (1.0, 3.0), 8.0, 1.0,
            '候选完全图  K_{|V_sub|}  （所有节点对都视为候选边）',
            fc='#F4F7FB', ec='#3A5BAA', fs=BOX_FS)

    # 截断 2：max_links
    add_box(ax, (1.0, 1.5), 8.0, 1.0,
            '【截断 ②】 max_links  →  保留 logit 最高的前 M 条候选边',
            fc='#FCEEF1', ec='#C0392B', fs=BOX_FS, fw='bold', tc='#7C2C20')

    # 阶段 5：最终学习子图
    add_box(ax, (2.6, 0.0), 4.8, 1.0, '最终学习子图  G_sub',
            fc='#EFFAEF', ec='#2E8B57', fs=BOX_FS, fw='bold', tc='#1F5E3D')

    # 主流程箭头
    chain_y = [(9.4, 8.7), (7.5, 7.0), (6.0, 5.5), (4.5, 4.0),
               (3.0, 2.5), (1.5, 1.0)]
    for y_top, y_bot in chain_y:
        add_arrow(ax, (5.0, y_top), (5.0, y_bot), color='#3A5BAA', lw=1.6)

    # 不再重复添加侧标，红色截断框已足够表达


def draw_right(ax):
    """右：参数入口"""
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.4, 11)
    ax.axis('off')
    ax.set_title('（b）参数入口：三个维度参数的链式依赖',
                 fontsize=15, fontweight='bold', color='#0B1F4B', pad=8)

    BOX_FS = 12

    # 三个核心维度参数（顶部）
    add_box(ax, (0.3, 9.3), 2.9, 1.1, 'emb_dim',
            fc='#FCEEF1', ec='#C0392B', fs=BOX_FS + 1, fw='bold', tc='#7C2C20')
    add_box(ax, (3.55, 9.3), 2.9, 1.1, 'gsl_rel_emb_dim',
            fc='#FCEEF1', ec='#C0392B', fs=BOX_FS + 1, fw='bold', tc='#7C2C20')
    add_box(ax, (6.8, 9.3), 2.9, 1.1, 'MLP_hidden_dim',
            fc='#FCEEF1', ec='#C0392B', fs=BOX_FS + 1, fw='bold', tc='#7C2C20')

    # 中间：受影响的模块
    add_box(ax, (0.3, 7.5), 2.9, 1.2,
            'GraphSAGE\n节点 / 关系 embedding',
            fc='#EAF2FB', ec='#1F49D8', fs=BOX_FS)
    add_box(ax, (3.55, 7.5), 2.9, 1.2,
            'GSL 模块\n关系编码、EdgeGate 输入',
            fc='#EAF2FB', ec='#1F49D8', fs=BOX_FS)
    add_box(ax, (6.8, 7.5), 2.9, 1.2,
            'Classifier\nMLP 隐藏层宽度',
            fc='#EAF2FB', ec='#1F49D8', fs=BOX_FS)

    for x in (1.75, 5.0, 8.25):
        add_arrow(ax, (x, 9.3), (x, 8.7), color='#C0392B', lw=1.6)

    # 第三层：特征张量维度
    add_box(ax, (0.3, 5.6), 2.9, 1.2,
            'h_v ∈ R^{emb_dim}\n（节点表示）',
            fc='#F4F7FB', ec='#3A5BAA', fs=BOX_FS)
    add_box(ax, (3.55, 5.6), 2.9, 1.2,
            'r_e ∈ R^{gsl_rel_emb_dim}\n（关系表示）',
            fc='#F4F7FB', ec='#3A5BAA', fs=BOX_FS)
    add_box(ax, (6.8, 5.6), 2.9, 1.2,
            'z ∈ R^{MLP_hidden_dim}\n（融合后特征）',
            fc='#F4F7FB', ec='#3A5BAA', fs=BOX_FS)

    for x in (1.75, 5.0, 8.25):
        add_arrow(ax, (x, 7.5), (x, 6.8), color='#3A5BAA', lw=1.4)

    # 链式依赖（横向箭头）
    add_arrow(ax, (3.2, 6.2), (3.55, 6.2), color='#E08600', lw=2.0)
    add_arrow(ax, (6.45, 6.2), (6.8, 6.2), color='#E08600', lw=2.0)

    # 第四层：拼接 → 分类器
    add_box(ax, (1.5, 3.6), 7.0, 1.3,
            'EdgeGate / 子图聚合：拼接 [h_u ‖ h_v ‖ r_e]\n输出维度 = 2·emb_dim + gsl_rel_emb_dim',
            fc='#FFF7E6', ec='#E08600', fs=BOX_FS, fw='bold', tc='#7A4F00')

    for x in (1.75, 5.0, 8.25):
        add_arrow(ax, (x, 5.6), (x, 4.9), color='#3A5BAA', lw=1.4)

    add_box(ax, (0.3, 1.5), 9.4, 1.5,
            'MLP Classifier：Linear(2·emb_dim + gsl_rel_emb_dim → MLP_hidden_dim)\n'
            '→ ReLU → Linear → 类别概率',
            fc='#EAF2FB', ec='#1F49D8', fs=BOX_FS)

    add_arrow(ax, (5.0, 3.6), (5.0, 3.0), color='#1F49D8', lw=1.6)

    # 链式总结（底部）
    add_box(ax, (0.3, -0.2), 9.4, 1.5,
            '链式压缩关系：emb_dim ↓  →  拼接维 ↓  →  MLP_hidden_dim ↓\n'
            '参数量 / 显存 / 推理时延 三者同步下降',
            fc='#EFFAEF', ec='#2E8B57', fs=BOX_FS, fw='bold', tc='#1F5E3D')

    add_arrow(ax, (5.0, 1.5), (5.0, 1.3), color='#2E8B57', lw=1.6)


def main():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 9.5),
                                     gridspec_kw=dict(wspace=0.06))
    draw_left(ax_l)
    draw_right(ax_r)

    # 顶部总标题
    fig.suptitle('图5-1  拓扑入口与参数入口：双维度压缩的截断位置与依赖链',
                 fontsize=17, fontweight='bold', color='#0B1F4B', y=0.985)

    # 底部统一图例
    handles = [
        mpatches.Patch(facecolor='#F4F7FB', edgecolor='#3A5BAA',
                       label='原 KnowDDI 流程节点'),
        mpatches.Patch(facecolor='#FCEEF1', edgecolor='#C0392B',
                       label='本文新增的压缩入口 / 维度参数'),
        mpatches.Patch(facecolor='#EAF2FB', edgecolor='#1F49D8',
                       label='受影响的模块'),
        mpatches.Patch(facecolor='#FFF7E6', edgecolor='#E08600',
                       label='中间张量拼接'),
        mpatches.Patch(facecolor='#EFFAEF', edgecolor='#2E8B57',
                       label='输出 / 链式压缩结论'),
    ]
    fig.legend(handles=handles, loc='lower center', fontsize=12,
               ncol=5, frameon=True, framealpha=0.92,
               edgecolor='#B7C0D4', bbox_to_anchor=(0.5, -0.005))

    out_path = os.path.join(os.path.dirname(__file__),
                            'fig5_1_dual_compression.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[完成] 图5-1 已保存至: {out_path}')


if __name__ == '__main__':
    main()
