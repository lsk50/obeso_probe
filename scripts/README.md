# scripts/ 目录说明

本目录存放 Obeso 管线复现的所有脚本。脚本按两位数字编号命名，编号大致反映执行顺序。

## 命名规则

- 格式：`NN_描述.py`（或 `.sh`）
- NN 为两位数字，从 00 到 99
- 描述用 snake_case，简明反映脚本功能
- 每个脚本应有 `--help` 参数说明用法
- 每个脚本开头写清楚：**输入路径、输出路径、关键超参、预期运行时长**

## 依赖

主要依赖：
- `transformers >= 4.40`
- `torch >= 2.1`（BF16 支持）
- `datasets`
- `scikit-learn`（AUC 计算）
- `peft`（LoRA 探针）
- `tqdm`, `numpy`, `pandas`
- `matplotlib`, `seaborn`（绘图）

安装命令：
```bash
pip install transformers torch datasets scikit-learn peft tqdm numpy pandas matplotlib seaborn
```

## 脚本清单（按编号）

见项目 README.md 第三节。
