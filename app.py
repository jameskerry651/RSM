"""RSM 响应面分析 — 交互式教学平台."""

import itertools

import streamlit as st
import numpy as np
import pandas as pd

from rsm.design import ExperimentDesigner, Factor
from rsm.model import RSMModel
from rsm.optimize import RSMOptimizer
from rsm.visualize import RSMVisualizer

st.set_page_config(page_title="RSM 响应面分析", page_icon="📐", layout="wide")

# ────────────────────── sidebar: navigation ──────────────────────

PAGES = {
    "🏠 概述与原理": "overview",
    "🧪 实验设计": "design",
    "📊 模型拟合": "fitting",
    "🎯 优化分析": "optimization",
    "🔬 完整案例": "case_study",
}

st.sidebar.title("RSM 响应面分析")
st.sidebar.markdown("---")
page = st.sidebar.radio("导航", list(PAGES.keys()), label_visibility="collapsed")
page_key = PAGES[page]


def _escape_latex_text(text: str) -> str:
    """转义文本中的 LaTeX 特殊字符，避免变量名渲染异常。"""
    replacements = {
        "\\": r"\backslash{}",
        "{": r"\{",
        "}": r"\}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "^": r"\^{}",
        "~": r"\~{}",
    }
    escaped = text
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def _build_equation_latex(model: RSMModel) -> str:
    """构建编码变量形式的 LaTeX 方程。"""
    coefs = np.asarray(model.coefficients, dtype=float)
    k = len(model._factor_names)
    x_syms = [f"x_{{{i+1}}}" for i in range(k)]

    terms = [f"{coefs[0]:+.4f}"]

    idx = 1
    for i in range(k):
        terms.append(f"{coefs[idx]:+.4f} \\, {x_syms[i]}")
        idx += 1

    if model.order == 2:
        for i in range(k):
            terms.append(f"{coefs[idx]:+.4f} \\, {x_syms[i]}^2")
            idx += 1

        for i in range(k):
            for j in range(i + 1, k):
                terms.append(f"{coefs[idx]:+.4f} \\, {x_syms[i]} {x_syms[j]}")
                idx += 1

    terms[0] = terms[0].lstrip("+")
    return " ".join(terms)


def _natural_equation_coefficients(model: RSMModel, factors: list[Factor]) -> tuple[float, np.ndarray, np.ndarray, dict[tuple[int, int], float]]:
    """将编码变量模型转换为自然变量模型系数。"""
    coefs = np.asarray(model.coefficients, dtype=float)
    scales = np.array([1.0 / f.half_range for f in factors], dtype=float)
    offsets = np.array([-f.center / f.half_range for f in factors], dtype=float)
    k = len(factors)

    const = float(coefs[0])
    linear = np.zeros(k, dtype=float)
    quadratic = np.zeros(k, dtype=float)
    interactions: dict[tuple[int, int], float] = {}

    idx = 1
    for i in range(k):
        beta = coefs[idx]
        const += beta * offsets[i]
        linear[i] += beta * scales[i]
        idx += 1

    if model.order == 2:
        for i in range(k):
            beta = coefs[idx]
            const += beta * offsets[i] ** 2
            linear[i] += 2 * beta * scales[i] * offsets[i]
            quadratic[i] += beta * scales[i] ** 2
            idx += 1

            for j in range(i + 1, k):
                beta = coefs[idx]
                const += beta * offsets[i] * offsets[j]
                linear[i] += beta * scales[i] * offsets[j]
                linear[j] += beta * offsets[i] * scales[j]
                interactions[(i, j)] = interactions.get((i, j), 0.0) + beta * scales[i] * scales[j]
                idx += 1

    return const, linear, quadratic, interactions


def _build_natural_equation_latex(model: RSMModel, factors: list[Factor]) -> str:
    """构建自然变量形式的 LaTeX 方程。"""
    const, linear, quadratic, interactions = _natural_equation_coefficients(model, factors)
    symbols = [rf"\text{{{_escape_latex_text(f.name)}}}" for f in factors]

    terms = [f"{const:+.4f}"]
    for i, symbol in enumerate(symbols):
        terms.append(f"{linear[i]:+.4f} \\, {symbol}")

    if model.order == 2:
        for i, symbol in enumerate(symbols):
            terms.append(f"{quadratic[i]:+.4f} \\, {symbol}^2")

        for i, symbol in enumerate(symbols):
            for j in range(i + 1, len(symbols)):
                terms.append(f"{interactions.get((i, j), 0.0):+.4f} \\, {symbol} \\cdot {symbols[j]}")

    terms[0] = terms[0].lstrip("+")
    return " ".join(terms)


def _build_coding_formula_latex(index: int, factor: Factor) -> str:
    """构建编码变量与自然变量的对应公式。"""
    name = _escape_latex_text(factor.name)
    return rf"x_{{{index + 1}}} = \frac{{\text{{{name}}} - {factor.center:.4f}}}{{{factor.half_range:.4f}}}"

# ══════════════════════════════════════════════════════════════════
#  PAGE 1 — 概述与原理
# ══════════════════════════════════════════════════════════════════
if page_key == "overview":
    st.title("响应面方法论 (Response Surface Methodology)")
    st.markdown("""
    ### 什么是 RSM？

    **响应面方法论** 是一组用于建立和探索因子与响应之间关系的数学和统计技术。
    它广泛用于工艺优化、产品设计和实验研究中。

    ### 核心思想

    RSM 通过有限次实验，拟合一个近似的数学模型（通常为二阶多项式），
    然后利用该模型来：

    1. **理解** 各因子对响应的影响
    2. **可视化** 响应面的形态
    3. **优化** 找到使响应最优的因子水平组合
    """)

    st.markdown("### 二阶响应面模型")
    st.latex(r"y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i=1}^{k} \beta_{ii} x_i^2 + \sum_{i<j} \beta_{ij} x_i x_j + \varepsilon")

    st.markdown("""
    其中:
    - $y$ : 响应变量
    - $x_i$ : 编码后的因子变量
    - $\\beta_0$ : 截距
    - $\\beta_i$ : 线性系数
    - $\\beta_{ii}$ : 二次系数（曲率效应）
    - $\\beta_{ij}$ : 交互作用系数
    - $\\varepsilon$ : 随机误差
    """)

    st.markdown("---")
    st.markdown("### RSM 工作流程")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### 1️⃣ 实验设计")
        st.info("选择合适的设计方案\n(CCD, BBD, 全因子)")
    with col2:
        st.markdown("#### 2️⃣ 收集数据")
        st.info("按设计矩阵执行实验\n获取响应值")
    with col3:
        st.markdown("#### 3️⃣ 模型拟合")
        st.info("回归分析, ANOVA\n残差诊断, 模型验证")
    with col4:
        st.markdown("#### 4️⃣ 优化")
        st.info("响应面可视化\n寻找最优条件")

    st.markdown("---")
    st.markdown("### 编码变量转换")
    st.markdown("为了使不同量纲的因子具有可比性，RSM 使用编码变量：")
    st.latex(r"x_i = \frac{X_i - \bar{X}_i}{\Delta X_i / 2}")
    st.markdown("其中 $X_i$ 为自然变量，$\\bar{X}_i$ 为中心点，$\\Delta X_i$ 为因子范围。")

    with st.expander("💡 互动练习：编码转换"):
        c1, c2 = st.columns(2)
        with c1:
            natural_low = st.number_input("低水平", value=20.0)
            natural_high = st.number_input("高水平", value=40.0)
        with c2:
            test_val = st.number_input("待转换的自然值", value=30.0)

        center = (natural_low + natural_high) / 2
        half_range = (natural_high - natural_low) / 2
        if half_range > 0:
            coded = (test_val - center) / half_range
            st.success(f"中心点 = {center}, 半范围 = {half_range}")
            st.success(f"编码值 = ({test_val} - {center}) / {half_range} = **{coded:.4f}**")

# ══════════════════════════════════════════════════════════════════
#  PAGE 2 — 实验设计
# ══════════════════════════════════════════════════════════════════
elif page_key == "design":
    st.title("🧪 实验设计")

    st.markdown("""
    实验设计是 RSM 的第一步。选择合适的设计方案可以在最少的实验次数下获得最多的信息。
    """)

    st.info("以下案例用于探究 DES-5 含水量、DES-5 摩尔比和超声时间对花椒多糖提取量的影响。你可以在下方直接编辑每个因子的三个水平（低水平 / 中水平 / 高水平）。")
    design_type = st.selectbox(
        "设计类型",
        ["指定水平全因子设计", "中心复合设计 (CCD)", "Box-Behnken 设计 (BBD)", "全因子设计"],
    )

    # ── 因子设定：三水平可编辑 ──
    st.markdown("---")
    st.subheader("因子设定（可编辑三水平）")

    default_factor_table = pd.DataFrame({
        "因子": ["DES-5 含水量", "DES-5 摩尔比", "超声时间"],
        "低水平": [10.0, 0.5, 20.0],
        "中水平": [30.0, 1.0, 60.0],
        "高水平": [50.0, 2.0, 100.0],
        "单位": ["%", "", "min"],
    })

    edited_factors = st.data_editor(
        default_factor_table,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "因子": st.column_config.TextColumn("因子名称", width="medium"),
            "低水平": st.column_config.NumberColumn("低水平 (编码 -1)", format="%.4f"),
            "中水平": st.column_config.NumberColumn("中水平 (编码 0)", format="%.4f"),
            "高水平": st.column_config.NumberColumn("高水平 (编码 +1)", format="%.4f"),
            "单位": st.column_config.TextColumn("单位", width="small"),
        },
        key="design_factor_editor",
    )

    factor_specs = edited_factors.to_dict("records")

    # 校验每个因子满足 低水平 < 中水平 < 高水平
    levels_ok = True
    for spec in factor_specs:
        low, center, high = spec["低水平"], spec["中水平"], spec["高水平"]
        if not (low < center < high):
            levels_ok = False
    if not levels_ok:
        st.warning("⚠️ 请确保每个因子的三个水平满足 **低水平 < 中水平 < 高水平**。")

    factors = [
        Factor(spec["因子"], spec["低水平"], spec["高水平"], spec["单位"])
        for spec in factor_specs
    ]
    factor_names = [f.name for f in factors]

    designer = ExperimentDesigner(factors)

    st.markdown("---")
    st.subheader("因子摘要")
    summary_rows = []
    for spec in factor_specs:
        summary_rows.append({
            "因子": spec["因子"],
            "低水平": spec["低水平"],
            "中水平": spec["中水平"],
            "高水平": spec["高水平"],
            "半范围": (spec["高水平"] - spec["低水平"]) / 2,
            "单位": spec["单位"],
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.subheader("实验水平")
    level_df = pd.DataFrame({
        "因素": [spec["因子"] for spec in factor_specs],
        "三水平": [
            f"{spec['低水平']:g} · {spec['中水平']:g} · {spec['高水平']:g} {spec['单位']}".strip()
            for spec in factor_specs
        ],
        "编码": ["-1 · 0 · +1"] * len(factor_specs),
    })
    st.dataframe(level_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("设计矩阵")

    try:
        if design_type == "指定水平全因子设计":
            # 三水平全因子设计：低(-1) / 中(0) / 高(+1)
            coded_choices = [-1.0, 0.0, 1.0]
            coded_rows = list(itertools.product(coded_choices, repeat=len(factor_specs)))
            code_matrix = np.array(coded_rows, dtype=float)

            def _natural_from_coded(spec: dict, coded: float) -> float:
                if abs(coded + 1.0) < 1e-9:
                    return spec["低水平"]
                if abs(coded) < 1e-9:
                    return spec["中水平"]
                return spec["高水平"]

            design_df = pd.DataFrame(
                [[_natural_from_coded(sp, c) for sp, c in zip(factor_specs, cr)] for cr in coded_rows],
                columns=factor_names,
            )
            for i, spec in enumerate(factor_specs):
                design_df[f"coded_{spec['因子']}"] = code_matrix[:, i]
        elif "CCD" in design_type:
            alpha_opt = st.selectbox("alpha 参数", ["orthogonal", "rotatable"])
            face_opt = st.selectbox("面类型", ["circumscribed", "inscribed", "faced"])
            design_df = designer.central_composite(alpha=alpha_opt, face=face_opt)
        elif "BBD" in design_type:
            design_df = designer.box_behnken()
        else:
            levels = st.slider("每因子水平数", 2, 5, 3)
            design_df = designer.full_factorial(levels)

        st.info(f"共 **{len(design_df)}** 组实验")
        st.dataframe(design_df.round(3), use_container_width=True, hide_index=True)

        csv = design_df.to_csv(index=False)
        st.download_button("📥 下载设计矩阵 (CSV)", csv, "design_matrix.csv", "text/csv")

        with st.expander("📖 设计类型说明"):
            if design_type == "指定水平全因子设计":
                st.markdown("""
                **指定水平全因子设计** 使用你在上方设定的三水平组合（低 / 中 / 高）：
                - 每个因子取 **3 个水平**（低水平 -1、中水平 0、高水平 +1）
                - 共 $3^{k}$ 组实验（当前 $k = %d$，即 $3 \\times 3 \\times 3 = %d$ 组）
                """ % (len(factor_specs), 3 ** len(factor_specs)))
            elif "CCD" in design_type:
                st.markdown("""
                **中心复合设计 (Central Composite Design)** 由三部分组成：
                - **因子部分**：$2^k$ 全因子或部分因子设计点
                - **轴向点**：沿每个轴正负方向各一个点（距中心 $\\alpha$）
                - **中心点**：重复的中心实验点

                设计参数:
                - **circumscribed**: 轴向点在立方体外部 ($\\alpha > 1$)
                - **inscribed**: 轴向点在立方体内部
                - **faced**: 轴向点在面中心 ($\\alpha = 1$)
                """)
            elif "BBD" in design_type:
                st.markdown("""
                **Box-Behnken 设计** 的特点：
                - 三水平设计，每次只在两个因子的中点和端点取值
                - 不包含极端顶点（所有因子同时取最高/最低水平的组合）
                - 实验次数通常少于 CCD
                - 适合因子 ≥ 3 的情况
                """)
            else:
                st.markdown("""
                **全因子设计** 是最完整的设计：
                - 所有因子水平的完全组合
                - 实验次数 = $levels^k$（增长极快！）
                - 可估计所有主效应和交互作用
                """)
    except Exception as e:
        st.error(f"设计生成失败: {e}")

# ══════════════════════════════════════════════════════════════════
#  PAGE 3 — 模型拟合
# ══════════════════════════════════════════════════════════════════
elif page_key == "fitting":
    st.title("📊 模型拟合与诊断")

    tab_demo, tab_upload = st.tabs(["🎮 内置示例", "📂 上传数据"])

    with tab_demo:
        st.markdown("使用模拟数据探索模型拟合过程。")
        st.markdown("#### 调节真实模型参数（模拟数据生成器）")
        c1, c2, c3 = st.columns(3)
        with c1:
            b0 = st.slider("β₀ (截距)", -10.0, 10.0, 5.0, 0.5)
            b1 = st.slider("β₁ (x₁ 线性)", -5.0, 5.0, 2.0, 0.5)
            b2 = st.slider("β₂ (x₂ 线性)", -5.0, 5.0, -1.5, 0.5)
        with c2:
            b11 = st.slider("β₁₁ (x₁²)", -5.0, 5.0, -2.0, 0.5)
            b22 = st.slider("β₂₂ (x₂²)", -5.0, 5.0, -1.0, 0.5)
            b12 = st.slider("β₁₂ (x₁x₂)", -5.0, 5.0, 1.0, 0.5)
        with c3:
            noise_level = st.slider("噪声水平 σ", 0.0, 3.0, 0.5, 0.1)
            n_center = st.slider("中心点重复次数", 1, 8, 3)

        factors = [Factor("x₁", -1, 1), Factor("x₂", -1, 1)]
        designer = ExperimentDesigner(factors)
        design_df = designer.central_composite()

        center_points = pd.DataFrame({"x₁": [0.0] * n_center, "x₂": [0.0] * n_center,
                                       "coded_x₁": [0.0] * n_center, "coded_x₂": [0.0] * n_center})
        design_df = pd.concat([design_df, center_points], ignore_index=True)

        X = design_df[["coded_x₁", "coded_x₂"]].values
        np.random.seed(42)
        y_true = b0 + b1 * X[:, 0] + b2 * X[:, 1] + b11 * X[:, 0]**2 + b22 * X[:, 1]**2 + b12 * X[:, 0] * X[:, 1]
        y = y_true + np.random.normal(0, noise_level, len(y_true))
        design_df["y_响应值"] = y

    with tab_upload:
        st.markdown("上传自己的数据 (CSV格式，前几列为因子，最后一列为响应值)")
        uploaded = st.file_uploader("选择CSV文件", type=["csv"])
        if uploaded is not None:
            upload_df = pd.read_csv(uploaded)
            st.dataframe(upload_df.head())
            factor_cols = st.multiselect("选择因子列", upload_df.columns.tolist(), default=upload_df.columns[:-1].tolist())
            response_col = st.selectbox("选择响应列", upload_df.columns.tolist(), index=len(upload_df.columns) - 1)

            if factor_cols and response_col:
                factors = [Factor(c, upload_df[c].min(), upload_df[c].max()) for c in factor_cols]
                designer = ExperimentDesigner(factors)
                X = designer.encode_df(upload_df[factor_cols])
                y = upload_df[response_col].values
                design_df = upload_df.copy()

    st.markdown("---")
    st.subheader("数据预览")
    if "design_df" in dir() and design_df is not None:
        st.dataframe(design_df.round(4), use_container_width=True, hide_index=True)

    if "X" in dir() and "y" in dir():
        order = st.radio("模型阶次", [1, 2], index=1, horizontal=True)
        model = RSMModel(order=order)
        model.fit(X, y, factor_names=[f.name for f in factors])

        st.markdown("---")
        st.subheader("回归系数")
        coef_df = model.coefficient_table()
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{model.r_squared:.4f}")
        c2.metric("Adj R²", f"{model.adj_r_squared:.4f}")
        c3.metric("RMSE", f"{model.rmse:.4f}")
        c4.metric("样本数", model.n)

        st.markdown("---")
        st.subheader("方差分析 (ANOVA)")
        anova_summary_df = model.anova_table(detail="summary")
        anova_term_df = model.anova_table(detail="terms")
        anova_tab1, anova_tab2 = st.tabs(["整体 ANOVA", "按因子/项 ANOVA"])
        with anova_tab1:
            st.dataframe(anova_summary_df, use_container_width=True, hide_index=True)
        with anova_tab2:
            st.caption("按因子/项表使用顺序平方和，展示各线性项、二次项和交互项对模型解释度的贡献。")
            st.dataframe(anova_term_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("残差诊断")
        viz = RSMVisualizer(model, factor_names=[f.name for f in factors])
        st.plotly_chart(viz.residual_diagnostics(), use_container_width=True)

        st.markdown("---")
        st.subheader("响应面可视化")
        if len(factors) >= 2:
            vc1, vc2 = st.columns(2)
            with vc1:
                st.plotly_chart(viz.surface_3d(0, 1), use_container_width=True)
            with vc2:
                st.plotly_chart(viz.contour(0, 1), use_container_width=True)

        st.subheader("扰动图与交互作用图")
        vc3, vc4 = st.columns(2)
        with vc3:
            st.plotly_chart(viz.perturbation_plot(), use_container_width=True)
        with vc4:
            if len(factors) >= 2:
                st.plotly_chart(viz.interaction_plot(0, 1), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE 4 — 优化分析
# ══════════════════════════════════════════════════════════════════
elif page_key == "optimization":
    st.title("🎯 优化分析")

    st.markdown("""
    在拟合好响应面模型后，我们可以利用数学优化方法在实验区域内寻找使响应达到最大（或最小）的因子组合。
    """)

    st.subheader("设定优化问题")
    c1, c2 = st.columns(2)
    with c1:
        n_factors_opt = st.slider("因子数", 2, 4, 2, key="opt_nf")
        opt_goal = st.radio("优化目标", ["最大化", "最小化"], horizontal=True)
    with c2:
        coded_lb = st.number_input("编码值下界", value=-1.5)
        coded_ub = st.number_input("编码值上界", value=1.5)

    factors = [Factor(f"x{i+1}", -1, 1) for i in range(n_factors_opt)]
    designer = ExperimentDesigner(factors)

    st.markdown("#### 模拟真实模型参数")
    coef_cols = st.columns(3)
    with coef_cols[0]:
        ob0 = st.number_input("β₀", value=80.0, key="ob0")
    params = {"b0": ob0}
    betas_linear = []
    betas_quad = []
    betas_inter = []
    with coef_cols[1]:
        for i in range(n_factors_opt):
            v = st.number_input(f"β_{i+1}", value=[3.0, -2.0, 1.0, 0.5][i], key=f"ob{i+1}")
            betas_linear.append(v)
    with coef_cols[2]:
        for i in range(n_factors_opt):
            v = st.number_input(f"β_{i+1}{i+1}", value=[-4.0, -3.0, -2.0, -1.5][i], key=f"ob{i+1}{i+1}")
            betas_quad.append(v)
    for i in range(n_factors_opt):
        for j in range(i + 1, n_factors_opt):
            v = st.number_input(f"β_{i+1}{j+1}", value=1.0, key=f"ob{i+1}{j+1}")
            betas_inter.append(v)

    if n_factors_opt >= 3:
        design_df = designer.box_behnken()
    else:
        design_df = designer.central_composite()

    coded_cols = [c for c in design_df.columns if c.startswith("coded_")]
    X = design_df[coded_cols].values
    y = np.full(len(X), ob0)
    for i in range(n_factors_opt):
        y = y + betas_linear[i] * X[:, i] + betas_quad[i] * X[:, i]**2
    inter_idx = 0
    for i in range(n_factors_opt):
        for j in range(i + 1, n_factors_opt):
            y = y + betas_inter[inter_idx] * X[:, i] * X[:, j]
            inter_idx += 1
    np.random.seed(123)
    y = y + np.random.normal(0, 0.3, len(y))

    model = RSMModel(order=2)
    model.fit(X, y, factor_names=[f.name for f in factors])
    optimizer = RSMOptimizer(model, factors)

    if st.button("🚀 运行优化", type="primary"):
        result = optimizer.optimize(maximize=(opt_goal == "最大化"), coded_bounds=(coded_lb, coded_ub))
        canonical = optimizer.canonical_analysis()

        st.markdown("---")
        st.subheader("优化结果")
        res_cols = st.columns(n_factors_opt + 1)
        for i, f in enumerate(factors):
            res_cols[i].metric(f.name, f"{result.optimal_coded[i]:.4f}", delta=f"自然值: {result.optimal_natural[i]:.4f}")
        res_cols[-1].metric("预测响应", f"{result.predicted_response:.4f}")

        st.markdown("---")
        st.subheader("典型分析")
        ca_c1, ca_c2 = st.columns(2)
        with ca_c1:
            st.markdown(f"**曲面形态**: {canonical['曲面形态']}")
            st.markdown(f"**驻点预测值**: {canonical['驻点预测值']:.4f}")
            sp_df = pd.DataFrame({
                "因子": [f.name for f in factors],
                "驻点(编码)": canonical["驻点(编码)"],
                "驻点(自然)": canonical["驻点(自然)"],
            })
            st.dataframe(sp_df, hide_index=True)
        with ca_c2:
            st.markdown("**特征值**:")
            eig_df = pd.DataFrame({
                "特征值": canonical["特征值"],
                "含义": ["凹(极大方向)" if e < 0 else "凸(极小方向)" for e in canonical["特征值"]],
            })
            st.dataframe(eig_df, hide_index=True)

        st.markdown("---")
        st.subheader("可视化")
        viz = RSMVisualizer(model, factor_names=[f.name for f in factors])
        if n_factors_opt >= 2:
            vis_c1, vis_c2 = st.columns(2)
            with vis_c1:
                st.plotly_chart(viz.surface_3d(0, 1), use_container_width=True)
            with vis_c2:
                st.plotly_chart(viz.contour(0, 1), use_container_width=True)
        st.plotly_chart(viz.perturbation_plot(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE 5 — 完整案例
# ══════════════════════════════════════════════════════════════════
elif page_key == "case_study":
    st.title("🔬 完整案例：酶催化反应优化")

    st.markdown("""
    ### 背景
    某生物工程实验室需要优化酶催化反应条件，以最大化产率。
    你可以在下方选择 **3 因子** 或 **4 因子** 的完整案例，并自定义每个因子的范围。
    """)

    n_case_factors = st.radio("案例因子数", [3, 4], horizontal=True, key="cs_nf")
    default_case_factors = [
        ("温度", 30.0, 50.0, "°C"),
        ("pH", 5.0, 9.0, ""),
        ("时间", 20.0, 60.0, "min"),
        ("底物浓度", 1.0, 5.0, "g/L"),
    ]

    # ── 步骤 1：用户自定义因子 ──
    st.markdown("### 步骤 1：因子范围确定")
    st.markdown("根据实际情况设定各因子的范围：")

    factor_cols = st.columns(n_case_factors)
    factors = []
    for i in range(n_case_factors):
        default_name, default_low, default_high, default_unit = default_case_factors[i]
        with factor_cols[i]:
            fname = st.text_input(f"因子{i+1} 名称", value=default_name, key=f"cs_f{i+1}")
            flow = st.number_input(f"因子{i+1} 低水平", value=default_low, key=f"cs_f{i+1}l")
            fhigh = st.number_input(f"因子{i+1} 高水平", value=default_high, key=f"cs_f{i+1}h")
            funit = st.text_input(f"因子{i+1} 单位", value=default_unit, key=f"cs_f{i+1}u")
            factors.append(Factor(fname, flow, fhigh, funit))

    factor_names = [f.name for f in factors]
    designer = ExperimentDesigner(factors)
    st.dataframe(designer.summary(), use_container_width=True, hide_index=True)

    # ── 步骤 2：CCD 设计 (旋转性 + circumscribed) ──
    st.markdown("### 步骤 2：中心复合设计 (CCD)")
    st.markdown("采用 **旋转性 (rotatable)** + **circumscribed** 类型的中心复合设计：")
    design_df = designer.central_composite(alpha="rotatable", face="circumscribed")
    coded_cols = [c for c in design_df.columns if c.startswith("coded_")]

    st.info(f"CCD 设计共 **{len(design_df)}** 组实验（含因子点 + 轴向点 + 中心点）")
    st.dataframe(design_df.round(4), use_container_width=True, hide_index=True)

    # ── 用户输入响应值 ──
    st.markdown("### 步骤 2.5：输入实验响应值")
    st.markdown("在下方表格中填入每组实验的实际响应值（产率 %）：")

    input_method = st.radio("数据输入方式", ["手动逐一输入", "批量粘贴", "使用模拟数据"], horizontal=True)

    y = None

    if input_method == "手动逐一输入":
        st.markdown("逐行输入每组实验的响应值：")
        response_values = []
        n_runs = len(design_df)
        n_cols_input = 4
        rows_per_col = (n_runs + n_cols_input - 1) // n_cols_input
        input_cols = st.columns(n_cols_input)
        for i in range(n_runs):
            col_idx = i // rows_per_col
            with input_cols[col_idx]:
                natural_vals = ", ".join(f"{design_df.iloc[i][fn]:.1f}" for fn in factor_names)
                val = st.number_input(
                    f"#{i+1} ({natural_vals})",
                    value=0.0, format="%.2f", key=f"cs_y_{i}",
                )
                response_values.append(val)
        y = np.array(response_values)
        if np.all(y == 0):
            st.warning("所有响应值均为 0，请输入实验数据后再进行分析。")
            y = None

    elif input_method == "批量粘贴":
        st.markdown("将响应值按顺序粘贴（每行一个数值，或用逗号分隔）：")
        raw_text = st.text_area(
            "响应值", height=200,
            placeholder=f"输入 {len(design_df)} 个数值，每行一个或逗号分隔",
            key="cs_batch",
        )
        if raw_text.strip():
            try:
                values = [float(v.strip()) for v in raw_text.replace("\n", ",").split(",") if v.strip()]
                if len(values) != len(design_df):
                    st.error(f"需要 {len(design_df)} 个数值，实际输入了 {len(values)} 个。")
                else:
                    y = np.array(values)
            except ValueError:
                st.error("输入格式错误，请确保每个值为数字。")

    else:  # 使用模拟数据
        st.markdown("使用内置模拟数据（用于教学演示）：")
        X_sim = design_df[coded_cols].values
        np.random.seed(2024)
        linear_coefs = np.array([4.5, -2.1, 1.8, 2.4][:n_case_factors], dtype=float)
        quadratic_coefs = np.array([-6.2, -4.8, -3.5, -2.9][:n_case_factors], dtype=float)
        interaction_coefs = {
            (0, 1): 1.5,
            (0, 2): -0.8,
            (1, 2): 0.5,
            (0, 3): 1.2,
            (1, 3): -0.6,
            (2, 3): 0.7,
        }

        y = np.full(len(X_sim), 75.0)
        for i in range(n_case_factors):
            y += linear_coefs[i] * X_sim[:, i] + quadratic_coefs[i] * X_sim[:, i] ** 2
        for (i, j), coef in interaction_coefs.items():
            if i < n_case_factors and j < n_case_factors:
                y += coef * X_sim[:, i] * X_sim[:, j]
        y += np.random.normal(0, 0.6, len(X_sim))
        st.success("已加载模拟数据。")

    if y is not None:
        design_df["响应值"] = y
        st.markdown("**完整实验数据：**")
        st.dataframe(design_df.round(4), use_container_width=True, hide_index=True)

        csv = design_df.to_csv(index=False)
        st.download_button("📥 下载实验数据 (CSV)", csv, "experiment_data.csv", "text/csv")

        # ── 步骤 3：模型拟合 ──
        st.markdown("### 步骤 3：模型拟合")
        X = design_df[coded_cols].values
        model = RSMModel(order=2)
        model.fit(X, y, factor_names=factor_names)

        # 渲染拟合公式
        st.markdown("#### 拟合回归方程")
        eq_coded = _build_equation_latex(model)
        eq_natural = _build_natural_equation_latex(model, factors)

        st.markdown("**编码变量形式：**")
        st.latex(f"Y = {eq_coded}")

        st.markdown("**编码关系：**")
        for i, factor in enumerate(factors):
            st.latex(_build_coding_formula_latex(i, factor))

        st.markdown("**自然变量形式：**")
        st.latex(f"Y = {eq_natural}")

        st.markdown("---")

        coef_df = model.coefficient_table()
        st.markdown("#### 回归系数表")
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("R²", f"{model.r_squared:.4f}")
        mc2.metric("Adj R²", f"{model.adj_r_squared:.4f}")
        mc3.metric("RMSE", f"{model.rmse:.4f}")

        st.markdown("**ANOVA 表**")
        anova_summary_df = model.anova_table(detail="summary")
        anova_term_df = model.anova_table(detail="terms")
        anova_tab1, anova_tab2 = st.tabs(["整体 ANOVA", "按因子/项 ANOVA"])
        with anova_tab1:
            st.dataframe(anova_summary_df, use_container_width=True, hide_index=True)
        with anova_tab2:
            st.caption("按因子/项表使用顺序平方和，便于比较各因子主效应、曲率效应和交互作用的贡献。")
            st.dataframe(anova_term_df, use_container_width=True, hide_index=True)

        # ── 步骤 4：残差诊断 ──
        st.markdown("### 步骤 4：残差诊断")
        viz = RSMVisualizer(model, factor_names=factor_names)
        st.plotly_chart(viz.residual_diagnostics(), use_container_width=True)

        # ── 步骤 5：响应面可视化 ──
        st.markdown("### 步骤 5：响应面可视化")
        pairs = [f"{factor_names[i]} × {factor_names[j]}" for i in range(n_case_factors) for j in range(i + 1, n_case_factors)]
        pair_indices = [(i, j) for i in range(n_case_factors) for j in range(i + 1, n_case_factors)]
        pair_choice = st.selectbox("选择因子对", pairs)
        sel_idx = pairs.index(pair_choice)
        xi, yi = pair_indices[sel_idx]
        other_indices = [i for i in range(n_case_factors) if i != xi and i != yi]
        hold_values = {}
        if other_indices:
            st.markdown("**固定其余因子的编码值：**")
            hold_cols = st.columns(len(other_indices))
            for col, other_idx in zip(hold_cols, other_indices):
                with col:
                    hold_values[other_idx] = st.slider(
                        f"{factor_names[other_idx]}",
                        min_value=-1.5, max_value=1.5, value=0.0, step=0.1, key=f"cs_hold_{other_idx}",
                    )

        vc1, vc2 = st.columns(2)
        with vc1:
            st.plotly_chart(viz.surface_3d(xi, yi, hold_values=hold_values), use_container_width=True)
        with vc2:
            st.plotly_chart(viz.contour(xi, yi, hold_values=hold_values), use_container_width=True)

        pc1, pc2 = st.columns(2)
        with pc1:
            st.plotly_chart(viz.perturbation_plot(), use_container_width=True)
        with pc2:
            st.plotly_chart(viz.interaction_plot(xi, yi), use_container_width=True)

        # ── 步骤 6：优化分析 ──
        st.markdown("### 步骤 6：优化分析")
        opt_goal_cs = st.radio("优化目标", ["最大化", "最小化"], horizontal=True, key="cs_opt_goal")
        optimizer = RSMOptimizer(model, factors)
        result = optimizer.optimize(maximize=(opt_goal_cs == "最大化"))
        canonical = optimizer.canonical_analysis()

        st.success(f"**曲面形态**: {canonical['曲面形态']}")
        opt_cols = st.columns(len(factors) + 1)
        for i, f in enumerate(factors):
            opt_cols[i].metric(
                f"{f.name} 最优值",
                f"{result.optimal_natural[i]:.2f} {f.unit}",
                delta=f"编码: {result.optimal_coded[i]:.3f}",
            )
        opt_cols[-1].metric("预测最优响应", f"{result.predicted_response:.2f}")

        st.markdown("---")
        st.markdown("""
        ### 结论
        通过 RSM 分析，我们可以：
        - 从回归方程的系数大小判断各因子影响程度
        - 从二次项系数的正负判断因子效应的曲率方向
        - 从交互项系数判断因子间是否存在协同或拮抗作用
        - 利用优化结果确定最佳实验条件
        """)

st.markdown("---")
st.caption("Designed for XiaoMin")
