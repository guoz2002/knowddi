# -*- coding: utf-8 -*-
"""图5-3 推理时延测量段时序示意 + 显存峰值 reset/read 配对关系图。
   —— 上：单个 batch 内的延迟测量段时序 ——
   —— 下：reset_peak_memory_stats / max_memory_allocated 的配对窗口 ——"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC',
                                   'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    BOX_FS = 14
    HEADER_FS = 15
    TITLE_FS = 17

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 10),
                                         gridspec_kw=dict(hspace=0.45,
                                                          height_ratios=[1, 1]))

    # ============================================================
    # 上图：推理时延测量段时序
    # ============================================================
    ax = ax_top
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis('off')
    ax.set_title('（a）单个 batch 推理时延测量段时序：synchronize 包夹的 t0 → t1',
                 fontsize=HEADER_FS + 1, fontweight='bold',
                 color='#0B1F4B', pad=8, loc='left')

    # 五个阶段
    phases = [
        ('DataLoader\n取 batch', 0.4, 1.6, '#F4F7FB', '#3A5BAA', 'cpu'),
        ('torch.cuda.\nsynchronize ①',  2.0, 1.0, '#FFF7E6', '#E08600', 'sync'),
        ('GraphSAGE → GSL → Classifier\n（GPU 前向）', 3.0, 4.4, '#EAF2FB', '#1F49D8', 'gpu'),
        ('指标累计\n（CPU 侧）', 7.4, 1.6, '#F4F7FB', '#3A5BAA', 'cpu'),
        ('torch.cuda.\nsynchronize ②', 9.0, 1.0, '#FFF7E6', '#E08600', 'sync'),
    ]
    bar_y = 1.7
    bar_h = 1.1
    for label, x, w, fc, ec, kind in phases:
        box = FancyBboxPatch((x, bar_y), w, bar_h,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             facecolor=fc, edgecolor=ec, linewidth=1.6)
        ax.add_patch(box)
        ax.text(x + w / 2, bar_y + bar_h / 2, label, ha='center', va='center',
                fontsize=BOX_FS, color='#0B1F4B',
                fontweight='bold' if kind == 'sync' else 'normal')

    # 时间轴
    ax.annotate('', xy=(11.8, 1.1), xytext=(0.2, 1.1),
                arrowprops=dict(arrowstyle='->', color='#0B1F4B', lw=1.6))
    ax.text(6.0, 0.7, '时间轴 →', ha='center', va='center',
            fontsize=BOX_FS, color='#0B1F4B', fontweight='bold')

    # t0 / t1 标记
    t0_x = 2.5  # synchronize ① 中点
    t1_x = 9.5  # synchronize ② 中点
    for tx, label, color in [(t0_x, 't0 = perf_counter()', '#C0392B'),
                             (t1_x, 't1 = perf_counter()', '#C0392B')]:
        ax.plot([tx, tx], [bar_y - 0.1, bar_y + bar_h + 0.1],
                color=color, linewidth=2.0, linestyle='--')
        ax.text(tx, bar_y + bar_h + 0.45, label, ha='center', va='bottom',
                fontsize=BOX_FS, color=color, fontweight='bold')

    # 测量窗口高亮
    win = FancyBboxPatch((t0_x, bar_y - 0.05), t1_x - t0_x, bar_h + 0.1,
                         boxstyle="round,pad=0.0,rounding_size=0.0",
                         facecolor='#FCEEF1', edgecolor='none', alpha=0.35)
    ax.add_patch(win)

    # 测量公式
    ax.text(6.0, 4.05,
            '单 batch 推理时延 ≈ t1 - t0  →  累加得到 inference_total_latency（分钟）',
            ha='center', va='center', fontsize=BOX_FS + 1,
            color='#7C2C20', fontweight='bold')

    # 注解：synchronize 的作用
    ax.annotate('① 等待此前 GPU kernel 全部完成\n再读 t0（避免提前计时）',
                xy=(2.5, bar_y), xytext=(0.3, 0.05),
                fontsize=BOX_FS - 1, color='#E08600',
                arrowprops=dict(arrowstyle='->', color='#E08600', lw=1.2))
    ax.annotate('② 等待 GPU 前向真正落盘\n再读 t1（避免漏计 GPU 时间）',
                xy=(9.5, bar_y), xytext=(8.6, 0.05),
                fontsize=BOX_FS - 1, color='#E08600',
                arrowprops=dict(arrowstyle='->', color='#E08600', lw=1.2))

    # ============================================================
    # 下图：显存峰值 reset / read 配对关系
    # ============================================================
    ax = ax_bot
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)

    ax.set_title('（b）显存峰值测量：reset_peak_memory_stats 与 max_memory_allocated 的配对窗口',
                 fontsize=HEADER_FS + 1, fontweight='bold',
                 color='#0B1F4B', pad=8, loc='left')

    # 模拟显存曲线
    t = np.linspace(0, 12, 600)
    mem = np.zeros_like(t)
    # 起步 0.4 GiB
    mem += 0.4
    # DataLoader 阶段微增
    mem += 0.05 * np.clip((t - 0.5) / 1.0, 0, 1)
    # 前向（3.0 - 7.4）：上升 → 平台 → 下降
    forward_mask = (t > 3.0) & (t < 7.4)
    forward_t = t[forward_mask]
    rise = 1.0 - np.exp(-(forward_t - 3.0) / 0.6)
    decay = np.exp(-(forward_t - 6.5) / 0.4)
    decay = np.where(forward_t < 6.5, 1.0, decay)
    mem[forward_mask] += 1.5 * rise * decay
    # 指标累计：略高的小尾巴
    mem += 0.08 * ((t > 7.4) & (t < 9.0))
    # 加噪声
    rng = np.random.default_rng(3)
    mem += rng.normal(0, 0.015, size=mem.shape)
    mem = np.clip(mem, 0.35, None)

    # 绘曲线
    ax.plot(t, mem + 0.5, color='#1F49D8', linewidth=2.3,
            label='当前已分配显存（GiB）')

    # reset / read 两条竖线
    reset_x = 0.6
    read_x = 11.4
    ax.plot([reset_x, reset_x], [0.4, 4.5], color='#2E8B57',
            linestyle='-', linewidth=2.2)
    ax.plot([read_x, read_x], [0.4, 4.5], color='#C0392B',
            linestyle='-', linewidth=2.2)

    ax.text(reset_x, 4.55, '① reset_peak_memory_stats()',
            ha='left', va='bottom', fontsize=BOX_FS,
            color='#2E8B57', fontweight='bold')
    ax.text(read_x, 4.55, '② max_memory_allocated()',
            ha='right', va='bottom', fontsize=BOX_FS,
            color='#C0392B', fontweight='bold')

    # 测量窗口高亮
    ax.axvspan(reset_x, read_x, color='#EFFAEF', alpha=0.45, zorder=0)

    # 峰值标记
    peak_idx = int(np.argmax(mem))
    peak_t = t[peak_idx]
    peak_v = mem[peak_idx] + 0.5
    ax.plot(peak_t, peak_v, marker='o', markersize=10,
            markerfacecolor='#FFF7E6', markeredgecolor='#E08600',
            markeredgewidth=2.0, zorder=5)
    ax.annotate(f'峰值显存\n（max_memory_allocated 读到 {peak_v:.2f} GiB）',
                xy=(peak_t, peak_v), xytext=(peak_t + 1.6, peak_v + 0.4),
                fontsize=BOX_FS, color='#E08600', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E08600', lw=1.4))

    # 阶段标注（淡色）
    stage_marks = [(0.5, 2.5, 'DataLoader 取 batch'),
                   (2.5, 3.0, 'sync ①'),
                   (3.0, 7.4, 'GPU 前向（GraphSAGE / GSL / Classifier）'),
                   (7.4, 9.0, '指标累计'),
                   (9.0, 9.6, 'sync ②')]
    for x0, x1, lab in stage_marks:
        ax.text((x0 + x1) / 2, 0.15, lab, ha='center', va='center',
                fontsize=BOX_FS - 2, color='#5A6B8C', style='italic')
        if x0 not in (0.5,):  # 起点已在 reset 标记内
            ax.plot([x0, x0], [0.35, 0.5], color='#B7C0D4',
                    linewidth=1.0, linestyle=':')

    # 坐标
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#7A8AA8')
    ax.set_xlabel('时间轴 →（与上图共享）', fontsize=BOX_FS,
                  color='#0B1F4B', labelpad=4)

    # 顶部总标题
    fig.suptitle('图5-3  推理时延测量段时序示意 与 显存峰值 reset/read 配对关系图',
                 fontsize=TITLE_FS, fontweight='bold', color='#0B1F4B',
                 y=0.985)

    # 底部图例
    handles = [
        mpatches.Patch(facecolor='#F4F7FB', edgecolor='#3A5BAA',
                       label='CPU 侧操作（DataLoader / 指标累计）'),
        mpatches.Patch(facecolor='#EAF2FB', edgecolor='#1F49D8',
                       label='GPU 前向（GraphSAGE / GSL / Classifier）'),
        mpatches.Patch(facecolor='#FFF7E6', edgecolor='#E08600',
                       label='torch.cuda.synchronize 同步点'),
        mpatches.Patch(facecolor='#FCEEF1', edgecolor='#C0392B',
                       label='时延测量窗口（t0 → t1）'),
        mpatches.Patch(facecolor='#EFFAEF', edgecolor='#2E8B57',
                       label='显存峰值测量窗口（reset → read）'),
    ]
    fig.legend(handles=handles, loc='lower center', fontsize=BOX_FS,
               ncol=3, frameon=True, framealpha=0.92,
               edgecolor='#B7C0D4', bbox_to_anchor=(0.5, -0.02))

    out_path = os.path.join(os.path.dirname(__file__),
                            'fig5_3_inference_timing.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[完成] 图5-3 已保存至: {out_path}')


if __name__ == '__main__':
    main()
