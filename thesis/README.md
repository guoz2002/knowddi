# thesis/ 目录索引

本目录为毕业论文生成目录，结构如下：

```
thesis/
├── generate_thesis.py          # 主生成脚本，运行后输出 Word 文档
├── README.md                   # 本索引文件
├── chapters/                   # 各章节内容模块
│   ├── ch01_introduction.py    # 第一章：绪论
│   ├── ch02_background.py      # 第二章：相关理论与技术基础
│   ├── ch03_baseline.py        # 第三章：KnowDDI模型分析与基准复现
│   ├── ch04_ablation.py        # 第四章：图结构学习机制的消融解耦研究
│   ├── ch05_lightweight.py     # 第五章：面向边缘部署的轻量化重构与评测
│   ├── ch06_conclusion.py      # 第六章：总结与展望
│   └── references_acknowledgement.py  # 参考文献与致谢
└── output/                     # 生成的 Word 文档输出目录
    └── 基于知识图谱增强的药物相互作用预测研究.docx
```

## 使用方法

```bash
cd /Users/gingersnap/project/knowddi/thesis
/opt/homebrew/bin/python3.11 generate_thesis.py
```

## 章节文件结构说明

每个章节文件（`ch0X_*.py`）包含：
- `CHAPTER_TITLE`：章节标题字符串
- `SECTIONS`：有序字典，键为节号（如 "1.1"），值为包含 `title` 和 `content` 的字典

`content` 字段支持：
- 普通正文段落（按换行分割）
- Markdown 表格（`|...|...|` 格式，自动转换为 Word 表格）
- 子节标题（`X.X.X 标题` 格式，自动识别为三级标题）

## 格式规范（依据山东科技大学本科毕业论文规范）

| 元素 | 字体 | 字号 | 对齐 | 其他 |
|------|------|------|------|------|
| 章标题 | 黑体 | 三号（16pt） | 居中 | 段前段后各12pt |
| 一级节标题 | 黑体 | 小三（14pt） | 左对齐 | 段前段后各6pt |
| 二/三级子节标题 | 黑体 | 四号（14pt）/小四（12pt） | 左对齐 | 段前段后各6pt |
| 正文 | 宋体/Times New Roman | 小四（12pt） | 两端对齐 | 首行缩进2字符，22pt固定行距 |
| 表内文字 | 宋体/Times New Roman | 五号（10.5pt） | 居中 | — |
| 参考文献 | 宋体/Times New Roman | 五号（10.5pt） | 两端对齐 | 悬挂缩进 |
| 页面设置 | — | — | — | A4，上3cm 下2.5cm 左3cm 右2.5cm |
