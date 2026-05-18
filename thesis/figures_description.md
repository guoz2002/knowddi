# 论文待插图 — 文字描述文档

> 本文档为论文中所有需要插入的图/示意图提供详细的文字描述，  
> 可据此手工绘制或使用 draw.io / Visio / PPT / Matplotlib 等工具出图。  
> 每节包含：图题、内容描述、布局建议、元素清单。

---

## 图1：KnowDDI 整体架构与数据流示意图

**图题**：图 3-1  KnowDDI 整体架构与数据流示意图  
**对应章节**：第3章 3.1.1

### 内容描述

该图应展示从原始输入到最终DDI预测的完整数据流，包含5个主要阶段，从左到右排列。

### 布局（横向流程图，5列）

```
[输入层]  →  [子图抽取]  →  [GraphSAGE编码]  →  [图结构学习]  →  [分类输出]
```

### 各阶段元素

**第1列：输入层**
- 矩形框：`背景知识图谱 BKG (Hetionet)`，内部画小节点网络示意
- 圆形：`目标药物对 (drug_h, drug_t)`，两个圆形节点用箭头相连，标注"待预测DDI"
- 说明：BKG包含药物、靶点、疾病、通路等多类实体

**第2列：子图抽取 (DIG)**
- 矩形框标题：`子图抽取模块 (DIG)`
- 内部说明：以 drug_h 和 drug_t 为种子，2跳邻域扩展
- 输出：一个小子图（约7-10个节点的局部网络），标注"局部子图 G_local"
- 参数标注（可选）：`hop=2, max_nodes_per_hop=10`

**第3列：全局编码 (GraphSAGE)**
- 矩形框标题：`GraphSAGE 全局编码模块`
- 说明：在完整 BKG 上预训练，生成初始节点嵌入
- 输出：矩阵/向量组，标注 `节点嵌入 H⁰ ∈ ℝ^{N×d}`
- 维度标注：`d = 32 (emb_dim)`

**第4列：图结构学习 (GSL)**
- 矩形框标题：`图结构学习模块 (GSL)`
- 内部分3行：
  - `① 候选完全图构造`
  - `② EdgeGateNetwork (去噪 + 补全)`
  - `③ 稀疏化 → 优化图结构`
- 输入：H⁰；输出：优化后节点表示 H*

**第5列：分类输出**
- 矩形框标题：`MLP 分类器`
- 输入：`[H*(drug_h) ‖ H*(drug_t) ‖ H*_graph]`（拼接符号‖）
- 输出：`DDI类型预测概率向量`
- 下方标注两种任务：
  - DrugBank：多分类（86类）→ 评价：Macro-F1 / ACC / κ
  - TWOSIDES：多标签（200类）→ 评价：AUROC / AUPRC

### 连接箭头
- 每个阶段之间用带箭头的粗线连接
- BKG → GraphSAGE 的箭头标注"全图预训练"
- BKG → DIG 的箭头标注"局部抽取"
- DIG → GSL 之间加入 H⁰（来自GraphSAGE）的斜向虚线箭头，标注"节点嵌入初始化"

### 配色建议
- 背景：白色
- 输入层：浅蓝色
- 子图抽取：浅绿色
- GraphSAGE：浅橙色
- GSL：浅紫色
- 分类器：浅红色/粉色

---

## 图2：DIG 子图抽取算法示意图

**图题**：图 3-2  有向子图提取（DIG）算法示意图  
**对应章节**：第3章 3.2.1

### 内容描述

该图通过一个具体示例说明 DIG 算法如何从大规模 BKG 中为目标药物对抽取局部子图。

### 布局（左右对比图，或分步流程图）

**推荐方案：分步三图（横排）**

**步骤1：背景知识图谱全局视图（左图）**
- 展示一个包含约20个节点的大图
- 节点类型用颜色区分：
  - 蓝色圆：Drug（药物）
  - 绿色圆：Target（靶点）
  - 橙色圆：Disease（疾病）
  - 紫色圆：Pathway（通路）
- 两个目标药物节点 drug_h 和 drug_t 用红色边框突出显示
- 图标题："原始背景知识图谱 BKG"

**步骤2：邻域扩展过程（中图）**
- 在背景图基础上，用不同深浅颜色表示扩展层次：
  - 红色：第0跳（drug_h, drug_t 本身）
  - 橙色：第1跳邻居
  - 黄色：第2跳邻居
- 用虚线圆圈圈出覆盖区域
- 图标题："2跳邻域扩展（hop=2）"

**步骤3：提取结果子图（右图）**
- 只保留被选中的节点（约8-12个）
- 节点之间保留原有边关系，加上关系标签（如"interacts_with", "targets"）
- drug_h 和 drug_t 用特殊形状（菱形）标注
- 图标题："提取的局部子图 G_local"

### 算法伪代码框（可选，放图下方）
```
输入: BKG, drug_h, drug_t, hop=2, max_nodes_per_hop=10
输出: G_local (局部子图)
1. queue = [drug_h, drug_t]; visited = {drug_h, drug_t}
2. for k in range(hop):
3.     new_nodes = []
4.     for node in queue:
5.         neighbors = sample(BKG.neighbors(node), max_nodes_per_hop)
6.         new_nodes.extend(neighbors - visited)
7.     visited.update(new_nodes); queue = new_nodes
8. G_local = BKG.subgraph(visited)
9. return G_local
```

---

## 图3：EdgeGateNetwork 结构示意图

**图题**：图 3-3  EdgeGateNetwork 结构示意图  
**对应章节**：第3章 3.2.3

### 内容描述

该图展示 EdgeGateNetwork 神经网络模块的内部结构，即图结构学习中核心的边权重计算网络。

### 布局（神经网络结构图，纵向）

**输入层（顶部）**
- 3个输入向量并排：
  - `src_feat`：源节点嵌入，维度 d
  - `rel_emb`：关系类型嵌入，维度 r
  - `dst_feat`：目标节点嵌入，维度 d
- 三个向量通过"拼接"操作合并：标注 `concat → ℝ^{2d+r}`

**隐层（中部）**
- 一个 MLP 模块：
  - 线性层：`Linear(2d+r, hidden_dim)`，ReLU激活
  - 线性层：`Linear(hidden_dim, hidden_dim)`，ReLU激活
- 标注：`hidden_dim = 64`（默认值）

**分叉输出头（底部，3路输出）**

左路 — `gate head`：
- `Linear(hidden_dim, 1)` → Sigmoid
- 输出：`gate_score ∈ (0,1)`（综合边权重）
- 说明：控制该边是否保留

中路 — `denoise head`：
- `Linear(hidden_dim, 1)` → Sigmoid
- 输出：`denoise_score ∈ (0,1)`（去噪得分）
- 说明：低分边被移除（抑制噪声边）

右路 — `completion head`：
- `Linear(hidden_dim, 1)` → Sigmoid
- 输出：`completion_score ∈ (0,1)`（补全得分）
- 说明：高分潜在边被激活（补全缺失边）

**最终融合（最底部）**
- 三路得分合并：`final_score = gate × (α·denoise + β·completion)`
- 输出：`边权重矩阵 W_edge`

### 配色
- 输入向量：浅蓝色矩形
- MLP层：灰色矩形
- 三路输出头：左-绿色（gate）、中-红色（denoise）、右-橙色（completion）

---

## 图4：DrugBank 消融实验各指标对比图

**图题**：图 4-1  DrugBank 数据集上四种消融变体的性能对比  
**对应章节**：第4章 4.2

### 内容描述

分组柱状图，展示4种消融变体在3个指标上的性能差异。

### 数据

| 变体 | Macro-F1 (%) | ACC (%) | Cohen's κ (%) |
|------|-------------|---------|--------------|
| Baseline（无GSL） | 91.49 | 92.86 | 91.53 |
| 仅去噪（Denoise-only） | 91.52 | 92.66 | 91.31 |
| 仅补全（Completion-only） | 90.00 | 92.89 | 91.57 |
| 去噪+补全（Full GSL） | 90.78 | 92.99 | 91.68 |

### 图表规格

- **图表类型**：分组柱状图（Grouped Bar Chart）
- **X轴**：3个指标组（Macro-F1 / ACC / Cohen's κ），每组4根柱子
- **Y轴**：百分比（%），范围建议 89.0% ~ 93.5%，便于看出差异
- **图例**：4种变体用4种颜色区分
  - Baseline：深蓝色
  - Denoise-only：橙色
  - Completion-only：绿色
  - Full GSL：红色
- 每根柱子顶部标注数值

### Python 绘图代码（参考）

```python
import matplotlib.pyplot as plt
import numpy as np

labels = ['Macro-F1 (%)', 'ACC (%)', "Cohen's κ (%)"]
baseline    = [91.49, 92.86, 91.53]
denoise     = [91.52, 92.66, 91.31]
completion  = [90.00, 92.89, 91.57]
full_gsl    = [90.78, 92.99, 91.68]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - 1.5*width, baseline,   width, label='Baseline',        color='#4472C4')
bars2 = ax.bar(x - 0.5*width, denoise,    width, label='Denoise-only',     color='#ED7D31')
bars3 = ax.bar(x + 0.5*width, completion, width, label='Completion-only',  color='#70AD47')
bars4 = ax.bar(x + 1.5*width, full_gsl,   width, label='Full GSL',         color='#FF0000')

ax.set_ylim(89.0, 93.8)
ax.set_ylabel('Score (%)')
ax.set_title('DrugBank Ablation Study')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.legend()
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        ax.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 2), textcoords='offset points', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('fig4-1_drugbank_ablation.png', dpi=300)
plt.show()
```

---

## 图5：TWOSIDES 消融实验各指标对比图

**图题**：图 4-2  TWOSIDES 数据集上四种消融变体的性能对比  
**对应章节**：第4章 4.3

### 数据

| 变体 | AUROC (%) | AUPRC (%) |
|------|-----------|-----------|
| Baseline（无GSL） | 95.44 | 94.11 |
| 仅去噪（Denoise-only） | 95.38 | 93.98 |
| 仅补全（Completion-only） | 95.41 | 94.05 |
| 去噪+补全（Full GSL） | 95.47 | 94.18 |

### 图表规格

- **图表类型**：分组柱状图
- **X轴**：2个指标组（AUROC / AUPRC），每组4根柱子
- **Y轴**：百分比（%），范围建议 93.5% ~ 96.0%
- **图例**：同图4-1，4种颜色对应4种变体

### Python 绘图代码（参考）

```python
import matplotlib.pyplot as plt
import numpy as np

labels = ['AUROC (%)', 'AUPRC (%)']
baseline    = [95.44, 94.11]
denoise     = [95.38, 93.98]
completion  = [95.41, 94.05]
full_gsl    = [95.47, 94.18]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - 1.5*width, baseline,   width, label='Baseline',        color='#4472C4')
ax.bar(x - 0.5*width, denoise,    width, label='Denoise-only',     color='#ED7D31')
ax.bar(x + 0.5*width, completion, width, label='Completion-only',  color='#70AD47')
ax.bar(x + 1.5*width, full_gsl,   width, label='Full GSL',         color='#FF0000')

ax.set_ylim(93.5, 96.0)
ax.set_ylabel('Score (%)')
ax.set_title('TWOSIDES Ablation Study')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.savefig('fig4-2_twosides_ablation.png', dpi=300)
plt.show()
```

---

## 图6：DrugBank 子图稀疏强度-精度/效率曲线图

**图题**：图 5-1  DrugBank 数据集上不同子图稀疏强度下的性能与效率变化  
**对应章节**：第5章 5.2

### 数据

| max_nodes_per_hop | Macro-F1 (%) | ACC (%) | 训练时长 | 显存占用 |
|---|---|---|---|---|
| 10（Baseline） | 91.49 | 92.86 | 3m12s | 1,124 MiB |
| 8 | 91.02 | 92.74 | 2m58s | 1,008 MiB |
| 6 | 90.51 | 92.61 | 2m44s | 967 MiB |
| 4 | 89.76 | 92.90 | 2m31s | 935 MiB |

### 图表规格

- **图表类型**：双Y轴折线图（左Y轴为性能，右Y轴为资源消耗）
- **X轴**：max_nodes_per_hop（4, 6, 8, 10），从右到左表示"稀疏化程度增加"
- **左Y轴**：Macro-F1 (%)，范围 88.0~92.5
- **右Y轴（次坐标轴）**：显存占用（MiB），范围 900~1200
- **4条线**：
  - 蓝色实线：Macro-F1
  - 绿色实线：ACC
  - 橙色虚线：训练时长（转换为秒：192, 178, 164, 151）
  - 红色虚线：显存占用（MiB）
- X轴右端（10）标注"Baseline"，X轴左端（4）标注"Sparse-only最优"

### Python 绘图代码（参考）

```python
import matplotlib.pyplot as plt
import numpy as np

x = [10, 8, 6, 4]  # max_nodes_per_hop
macro_f1 = [91.49, 91.02, 90.51, 89.76]
acc      = [92.86, 92.74, 92.61, 92.90]
time_s   = [192, 178, 164, 151]   # 训练时长（秒）
mem_mib  = [1124, 1008, 967, 935]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

ax1.plot(x, macro_f1, 'b-o', label='Macro-F1 (%)', linewidth=2)
ax1.plot(x, acc,      'g-s', label='ACC (%)',       linewidth=2)
ax2.plot(x, mem_mib,  'r--^', label='Memory (MiB)', linewidth=2)

ax1.set_xlabel('max_nodes_per_hop')
ax1.set_ylabel('Score (%)', color='black')
ax2.set_ylabel('Memory (MiB)', color='red')
ax1.set_ylim(88.0, 93.5)
ax2.set_ylim(800, 1300)
ax1.set_xticks(x)
ax1.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
ax1.text(10.05, 88.2, 'Baseline', fontsize=9, color='gray')
ax1.set_title('DrugBank: Subgraph Sparsity vs Performance/Memory')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
plt.tight_layout()
plt.savefig('fig5-1_drugbank_sparse.png', dpi=300)
plt.show()
```

---

## 图7：DrugBank 特征压缩强度-精度/效率曲线图

**图题**：图 5-2  DrugBank 数据集上不同特征压缩强度下的性能与效率变化  
**对应章节**：第5章 5.2

### 数据

| emb_dim | Macro-F1 (%) | ACC (%) | 训练时长 | 显存占用 |
|---|---|---|---|---|
| 32（Baseline） | 91.49 | 92.86 | 3m12s | 1,124 MiB |
| 24 | 91.15 | 92.77 | 2m31s | 956 MiB |
| 16 | 90.91 | 92.80 | 2m04s | 859 MiB |
| 8  | 89.43 | 92.48 | 1m52s | 782 MiB |

### 图表规格

- 同图5-1（双Y轴折线图），X轴为 emb_dim（8, 16, 24, 32）
- X轴右端（32）标注"Baseline"，X轴左端（8）标注"高压缩区"
- Feature-only 最优点在 emb_dim=16 处标注：`"Feature-only最优: 90.91%"`

### Python 绘图代码（参考）

```python
import matplotlib.pyplot as plt

x = [32, 24, 16, 8]  # emb_dim
macro_f1 = [91.49, 91.15, 90.91, 89.43]
acc      = [92.86, 92.77, 92.80, 92.48]
mem_mib  = [1124, 956, 859, 782]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

ax1.plot(x, macro_f1, 'b-o', label='Macro-F1 (%)', linewidth=2)
ax1.plot(x, acc,      'g-s', label='ACC (%)',       linewidth=2)
ax2.plot(x, mem_mib,  'r--^', label='Memory (MiB)', linewidth=2)

# 标注最优点
ax1.annotate('Feature-only 最优\n90.91%', xy=(16, 90.91),
             xytext=(20, 89.8), arrowprops=dict(arrowstyle='->'), fontsize=9)

ax1.set_xlabel('emb_dim')
ax1.set_ylabel('Score (%)')
ax2.set_ylabel('Memory (MiB)', color='red')
ax1.set_ylim(88.0, 93.5)
ax2.set_ylim(600, 1300)
ax1.set_xticks(x)
ax1.set_title('DrugBank: Feature Compression vs Performance/Memory')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
plt.tight_layout()
plt.savefig('fig5-2_drugbank_feature.png', dpi=300)
plt.show()
```

---

## 图8：DrugBank 轻量化方案综合对比图

**图题**：图 5-3  DrugBank 数据集上轻量化方案综合对比（相对于 Baseline 的变化率）  
**对应章节**：第5章 5.3

### 数据（相对于 Baseline 的变化率）

Baseline 基准：Macro-F1=91.49%, 训练时长=192s, 显存=1124MiB

| 方案 | Macro-F1变化率 | 训练时长变化率 | 显存变化率 |
|------|-------------|------------|---------|
| 仅稀疏化 (Sparse-only) | −1.89% | −21.4% | −16.8% |
| 仅特征压缩 (Feature-only) | −0.64% | −35.4% | −23.6% |
| 联合轻量化 (Joint) | −1.75% | −35.4% | −23.6% |

### 图表规格

**推荐方案：雷达图（Radar Chart）或分组气泡图**

#### 方案A：雷达图（展示多维权衡关系）

三个轴：
1. **精度保留率**（=1 - |Macro-F1变化率|）：越大越好，Sparse=98.11%, Feature=99.36%, Joint=98.25%
2. **训练速度提升率**（=|训练时长变化率|）：越大越好
3. **显存节省率**（=|显存变化率|）：越大越好

每种方案一条多边形，颜色区分：
- 蓝色多边形：Sparse-only
- 橙色多边形：Feature-only
- 绿色多边形：Joint

#### 方案B：分组柱状图（更直观）

X轴：3个方案；每组3根柱子（Macro-F1变化率、训练时长节省率、显存节省率）  
注意：Macro-F1是负数（精度损失），用负Y轴或取绝对值

### Python 绘图代码（雷达图，参考）

```python
import matplotlib.pyplot as plt
import numpy as np

categories = ['精度保留率(%)', '训练速度\n提升率(%)', '显存\n节省率(%)']
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

sparse  = [98.11, 21.4, 16.8];  sparse  += sparse[:1]
feature = [99.36, 35.4, 23.6];  feature += feature[:1]
joint   = [98.25, 35.4, 23.6];  joint   += joint[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, sparse,  'b-o', linewidth=2, label='Sparse-only')
ax.fill(angles, sparse,  alpha=0.1, color='blue')
ax.plot(angles, feature, 'r-s', linewidth=2, label='Feature-only')
ax.fill(angles, feature, alpha=0.1, color='red')
ax.plot(angles, joint,   'g-^', linewidth=2, label='Joint')
ax.fill(angles, joint,   alpha=0.1, color='green')

ax.set_thetagrids(np.degrees(angles[:-1]), categories)
ax.set_ylim(0, 100)
ax.set_title('DrugBank 轻量化方案综合对比', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('fig5-3_drugbank_comparison.png', dpi=300)
plt.show()
```

---

## 附录：出图工具建议

| 图编号 | 图类型 | 推荐工具 |
|--------|--------|---------|
| 图3-1（整体架构图） | 流程图/架构图 | draw.io（免费，推荐）或 Visio |
| 图3-2（DIG子图抽取） | 网络图+流程图 | draw.io 或 Gephi（网络可视化） |
| 图3-3（EdgeGateNetwork） | 神经网络结构图 | draw.io 或 NN-SVG（在线工具） |
| 图4-1（DrugBank消融） | 分组柱状图 | Python Matplotlib（代码已提供） |
| 图4-2（TWOSIDES消融） | 分组柱状图 | Python Matplotlib（代码已提供） |
| 图5-1（稀疏曲线） | 双Y轴折线图 | Python Matplotlib（代码已提供） |
| 图5-2（特征压缩曲线） | 双Y轴折线图 | Python Matplotlib（代码已提供） |
| 图5-3（综合对比） | 雷达图 | Python Matplotlib（代码已提供） |

### draw.io 快速上手
- 网址：https://app.diagrams.net
- 建议模板：Flowchart（流程图）或 Network（网络图）
- 导出时选择：Export as PNG，300 DPI，白色背景

### 论文插图格式要求
- 格式：PNG（位图）或 EPS/SVG（矢量图，推荐）
- 分辨率：≥300 DPI
- 字体：图内文字建议与正文保持一致（宋体/Times New Roman，10-12pt）
- 图题：置于图下方，格式"图X-X  标题"，居中，宋体10.5pt
