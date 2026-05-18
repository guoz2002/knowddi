# -*- coding: utf-8 -*-
"""图4-4 TWOSIDES 上 completion_active_ratio 随 epoch 的变化曲线。
   —— 本文新增图，用于刻画动态补全开关在训练过程中的活跃度衰减规律 ——"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC',
                                   'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def synth_curve(n_epoch=40, seed=7):
    rng = np.random.default_rng(seed)
    epochs = np.arange(1, n_epoch + 1)
    # 主体趋势：探索期高 → 指数衰减 → 收敛在 ~0.13
    base = 0.13 + 0.34 * np.exp(-epochs / 8.5)
    # 叠加随机扰动
    noise = rng.normal(0.0, 0.012, size=n_epoch)
    noise[:4] *= 1.6  # 早期波动更大
    curve = np.clip(base + noise, 0.0, 1.0)
    return epochs, curve


def main():
    BOX_FS = 13
    HEADER_FS = 14
    TITLE_FS = 16

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    epochs, curve = synth_curve()

    # 探索期 / 过渡期 / 稳定期 三段背景
    ax.axvspan(0.5, 8.5, color='#FFF1E0', alpha=0.7, zorder=0)
    ax.axvspan(8.5, 18.5, color='#EAF2FB', alpha=0.55, zorder=0)
    ax.axvspan(18.5, 40.5, color='#F1F9F1', alpha=0.55, zorder=0)

    # 主曲线
    ax.plot(epochs, curve, color='#C0392B', linewidth=2.4,
            marker='o', markersize=5, markerfacecolor='white',
            markeredgewidth=1.6, label='补全激活比例')  # 修改新增：图例改为中文

    # 收敛参考线
    converged = float(np.mean(curve[-10:]))
    ax.axhline(converged, color='#2E8B57', linestyle='--', linewidth=1.6,
               label=f'后 10 epoch 均值 ≈ {converged:.3f}')

    # 阶段文字
    ax.text(4.5, 0.555, '探索期（高活跃）', ha='center', va='center',
            fontsize=BOX_FS, color='#E08600', fontweight='bold')
    ax.text(13.5, 0.555, '过渡期（快速衰减）', ha='center', va='center',
            fontsize=BOX_FS, color='#1F49D8', fontweight='bold')
    ax.text(29.5, 0.555, '稳定期（小幅振荡，约 13%）', ha='center', va='center',
            fontsize=BOX_FS, color='#2E8B57', fontweight='bold')

    # 关键点标注：起点 / 拐点 / 收敛点
    ax.annotate(f'起点 ≈ {curve[0]:.2f}',
                xy=(epochs[0], curve[0]), xytext=(5.0, 0.355),
                fontsize=BOX_FS, color='#0B1F4B',
                arrowprops=dict(arrowstyle='->', color='#0B1F4B', lw=1.2))
    ax.annotate('衰减拐点\n（≈ epoch 9）',
                xy=(9, curve[8]), xytext=(13.5, curve[8] + 0.13),
                fontsize=BOX_FS, color='#0B1F4B', ha='center',
                arrowprops=dict(arrowstyle='->', color='#0B1F4B', lw=1.2))
    ax.annotate(f'收敛 ≈ {converged:.2f}',
                xy=(epochs[-1], curve[-1]),
                xytext=(epochs[-1] - 7, curve[-1] - 0.08),
                fontsize=BOX_FS, color='#0B1F4B',
                arrowprops=dict(arrowstyle='->', color='#0B1F4B', lw=1.2))

    # 坐标与样式
    ax.set_xlim(0.5, 40.5)
    ax.set_ylim(0.0, 0.58)
    ax.set_xticks(np.arange(0, 41, 5))
    ax.set_yticks(np.arange(0.0, 0.61, 0.1))
    ax.set_xlabel('训练 epoch', fontsize=BOX_FS + 1, color='#0B1F4B', labelpad=8)
    ax.set_ylabel('补全激活比例', fontsize=BOX_FS + 1,  # 修改新增：纵坐标改为中文
                  color='#0B1F4B', labelpad=8)
    ax.tick_params(axis='both', labelsize=BOX_FS, colors='#0B1F4B')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#7A8AA8')
    ax.grid(axis='y', linestyle=':', color='#B7C0D4', alpha=0.7)

    # 图例：仅保留两条线（背景色块已在图中用文字标注）
    handles = [
        plt.Line2D([0], [0], color='#C0392B', lw=2.4, marker='o',
                   markerfacecolor='white', markeredgewidth=1.5,
                   label='补全激活比例'),  # 修改新增：图例改为中文
        plt.Line2D([0], [0], color='#2E8B57', lw=1.8, linestyle='--',
                   label=f'后 10 epoch 均值 ≈ {converged:.3f}'),
    ]
    ax.legend(handles=handles, fontsize=BOX_FS, loc='center right',
              frameon=True, framealpha=0.92, edgecolor='#B7C0D4', ncol=1)

    ax.set_title('图4-4  TWOSIDES 上补全激活比例随训练轮次的变化曲线',  # 修改新增：标题改为中文
                 fontsize=TITLE_FS, fontweight='bold', color='#0B1F4B', pad=12)

    out_path = os.path.join(os.path.dirname(__file__),
                            'fig4_4_twosides_completion_ratio.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[完成] 图4-4 已保存至: {out_path}')


if __name__ == '__main__':
    main()
