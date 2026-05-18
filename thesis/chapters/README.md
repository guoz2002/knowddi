# chapters/ 目录索引

各章节文件均为标准 Python 模块，由 `generate_thesis.py` 动态加载。

| 文件名 | 对应章节 | 包含节数 |
|--------|----------|----------|
| ch01_introduction.py | 第一章：绪论 | 4节（1.1~1.4） |
| ch02_background.py | 第二章：相关理论与技术基础 | 5节（2.1~2.5） |
| ch03_baseline.py | 第三章：KnowDDI模型分析与基准复现 | 4节（3.1~3.4） |
| ch04_ablation.py | 第四章：图结构学习机制的消融解耦研究 | 4节（4.1~4.4） |
| ch05_lightweight.py | 第五章：面向边缘部署的轻量化重构与评测 | 4节（5.1~5.4） |
| ch06_conclusion.py | 第六章：总结与展望 | 3节（6.1~6.3） |
| references_acknowledgement.py | 参考文献与致谢 | — |

## 修改说明

如需修改某章节内容，直接编辑对应 `.py` 文件中的 `SECTIONS` 字典下的 `content` 字符串即可，无需修改生成脚本。

修改后重新运行：
```bash
/opt/homebrew/bin/python3.11 /Users/gingersnap/project/knowddi/thesis/generate_thesis.py
```
