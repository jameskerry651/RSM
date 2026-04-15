# RSM - 响应面分析教学工具

交互式响应面方法论 (Response Surface Methodology) 学习平台，涵盖实验设计、模型拟合、可视化与优化。

## 功能

- **实验设计**: 中心复合设计 (CCD)、Box-Behnken 设计、全因子设计
- **模型拟合**: 一阶/二阶多项式模型、ANOVA 分析、模型诊断
- **可视化**: 3D 响应曲面、等高线图、Perturbation 图、残差分析
- **优化**: 单目标/多目标优化、寻找最优参数组合
- **交互式教学**: 参数实时调节、步骤引导式学习

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 项目结构

```
RSM/
├── app.py                  # Streamlit 主应用
├── rsm/
│   ├── __init__.py
│   ├── design.py           # 实验设计模块
│   ├── model.py            # 模型拟合与诊断
│   ├── optimize.py         # 优化模块
│   └── visualize.py        # 可视化模块
├── examples/
│   └── demo_notebook.ipynb # Jupyter 教学笔记本
├── requirements.txt
└── README.md
```
