"""模型拟合模块 — 多项式回归、ANOVA、残差诊断."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures


@dataclass
class RSMModel:
    """二阶响应面模型: y = β₀ + Σβᵢxᵢ + Σβᵢᵢxᵢ² + Σβᵢⱼxᵢxⱼ + ε"""

    order: Literal[1, 2] = 2
    include_interactions: bool = True

    # fitted state
    coefficients: np.ndarray | None = field(default=None, repr=False)
    feature_names: list[str] = field(default_factory=list, repr=False)
    _raw_feature_names: list[str] = field(default_factory=list, repr=False)
    _X_design: np.ndarray | None = field(default=None, repr=False)
    _y: np.ndarray | None = field(default=None, repr=False)
    _factor_names: list[str] = field(default_factory=list, repr=False)

    def _build_features(self, X: np.ndarray) -> np.ndarray:
        interaction = self.include_interactions and self.order == 2
        poly = PolynomialFeatures(degree=self.order, interaction_only=not interaction, include_bias=True)
        return poly.fit_transform(X), poly.get_feature_names_out()

    def fit(self, X: np.ndarray, y: np.ndarray, factor_names: list[str] | None = None) -> "RSMModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._factor_names = factor_names or [f"x{i+1}" for i in range(X.shape[1])]

        X_design, names = self._build_features(X)
        self._raw_feature_names = list(names)
        self.feature_names = [_pretty_name(n, self._factor_names) for n in names]

        # OLS via normal equation with pseudo-inverse for stability
        self.coefficients = np.linalg.lstsq(X_design, y, rcond=None)[0]
        self._X_design = X_design
        self._y = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_design, _ = self._build_features(np.asarray(X, dtype=float))
        return X_design @ self.coefficients

    # ---- diagnostics ----

    @property
    def fitted_values(self) -> np.ndarray:
        return self._X_design @ self.coefficients

    @property
    def residuals(self) -> np.ndarray:
        return self._y - self.fitted_values

    @property
    def n(self) -> int:
        return len(self._y)

    @property
    def p(self) -> int:
        return len(self.coefficients)

    @property
    def dof_residual(self) -> int:
        return self.n - self.p

    @property
    def ss_total(self) -> float:
        return float(np.sum((self._y - self._y.mean()) ** 2))

    @property
    def ss_residual(self) -> float:
        return float(np.sum(self.residuals ** 2))

    @property
    def ss_regression(self) -> float:
        return self.ss_total - self.ss_residual

    @property
    def r_squared(self) -> float:
        return 1 - self.ss_residual / self.ss_total if self.ss_total > 0 else 0.0

    @property
    def adj_r_squared(self) -> float:
        n, p = self.n, self.p
        if n - p <= 0:
            return float("nan")
        return 1 - (self.ss_residual / (n - p)) / (self.ss_total / (n - 1))

    @property
    def mse(self) -> float:
        return self.ss_residual / max(self.dof_residual, 1)

    @property
    def rmse(self) -> float:
        return np.sqrt(self.mse)

    def anova_table(self, detail: Literal["summary", "terms"] = "summary") -> pd.DataFrame:
        if detail == "terms":
            return self._term_anova_table()

        dof_reg = self.p - 1
        dof_res = self.dof_residual
        ms_reg = self.ss_regression / max(dof_reg, 1)
        ms_res = self.mse
        f_value = ms_reg / ms_res if ms_res > 0 else float("inf")
        p_value = 1 - stats.f.cdf(f_value, dof_reg, dof_res) if dof_res > 0 else float("nan")

        return pd.DataFrame({
            "来源": ["回归", "残差", "总计"],
            "平方和": [self.ss_regression, self.ss_residual, self.ss_total],
            "自由度": [dof_reg, dof_res, self.n - 1],
            "均方": [ms_reg, ms_res, ""],
            "F值": [f_value, "", ""],
            "p值": [p_value, "", ""],
        })

    def _term_anova_table(self) -> pd.DataFrame:
        """按因子/项给出顺序平方和 ANOVA 表。"""
        H = self._X_design
        y = self._y
        mse_full = self.mse
        dof_res = self.dof_residual
        dof_reg = self.p - 1
        ss_reg = self.ss_regression

        ordered_terms = sorted(
            (_term_metadata(raw, self._factor_names, col_idx) for col_idx, raw in enumerate(self._raw_feature_names[1:], start=1)),
            key=lambda item: (item["sort_group"], item["factor_indices"]),
        )

        current_cols = [0]
        ss_prev = _ols_sse(H[:, current_cols], y)
        rows = []

        for term in ordered_terms:
            current_cols.append(term["column_index"])
            ss_curr = _ols_sse(H[:, current_cols], y)
            ss_term = max(ss_prev - ss_curr, 0.0)
            df_term = 1
            ms_term = ss_term / df_term
            f_value = ms_term / mse_full if mse_full > 0 else float("inf")
            p_value = 1 - stats.f.cdf(f_value, df_term, dof_res) if dof_res > 0 else float("nan")

            rows.append({
                "来源": term["label"],
                "类型": term["type_label"],
                "平方和": ss_term,
                "自由度": df_term,
                "均方": ms_term,
                "F值": f_value,
                "p值": p_value,
                "贡献率(%)": ss_term / ss_reg * 100 if ss_reg > 0 else float("nan"),
            })
            ss_prev = ss_curr

        ms_reg = ss_reg / max(dof_reg, 1)
        f_reg = ms_reg / mse_full if mse_full > 0 else float("inf")
        p_reg = 1 - stats.f.cdf(f_reg, dof_reg, dof_res) if dof_res > 0 else float("nan")

        rows.extend([
            {
                "来源": "回归",
                "类型": "整体",
                "平方和": ss_reg,
                "自由度": dof_reg,
                "均方": ms_reg,
                "F值": f_reg,
                "p值": p_reg,
                "贡献率(%)": 100.0 if ss_reg > 0 else float("nan"),
            },
            {
                "来源": "残差",
                "类型": "整体",
                "平方和": self.ss_residual,
                "自由度": dof_res,
                "均方": mse_full,
                "F值": float("nan"),
                "p值": float("nan"),
                "贡献率(%)": self.ss_residual / self.ss_total * 100 if self.ss_total > 0 else float("nan"),
            },
            {
                "来源": "总计",
                "类型": "整体",
                "平方和": self.ss_total,
                "自由度": self.n - 1,
                "均方": float("nan"),
                "F值": float("nan"),
                "p值": float("nan"),
                "贡献率(%)": 100.0 if self.ss_total > 0 else float("nan"),
            },
        ])

        return pd.DataFrame(rows)

    def coefficient_table(self) -> pd.DataFrame:
        mse = self.mse
        H = self._X_design
        try:
            cov = mse * np.linalg.pinv(H.T @ H)
        except np.linalg.LinAlgError:
            cov = mse * np.eye(H.shape[1])
        se = np.sqrt(np.abs(np.diag(cov)))
        t_vals = self.coefficients / np.where(se > 0, se, 1)
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), self.dof_residual))

        return pd.DataFrame({
            "项": self.feature_names,
            "系数": self.coefficients,
            "标准误": se,
            "t值": t_vals,
            "p值": p_vals,
            "显著": ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "" for p in p_vals],
        })

    def lack_of_fit_test(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> dict:
        """纯误差-失拟检验 (需要重复实验)."""
        y = np.asarray(y).ravel()
        y_hat = self.predict(X)

        if groups is None:
            rounded = np.round(np.asarray(X), decimals=6)
            groups = pd.DataFrame(rounded).apply(tuple, axis=1).values

        unique_groups = np.unique(groups)
        ss_pe = 0.0
        for g in unique_groups:
            mask = groups == g
            if mask.sum() > 1:
                ss_pe += np.sum((y[mask] - y[mask].mean()) ** 2)

        ss_lof = self.ss_residual - ss_pe
        dof_pe = sum(((groups == g).sum() - 1) for g in unique_groups if (groups == g).sum() > 1)
        dof_lof = self.dof_residual - dof_pe

        if dof_lof <= 0 or dof_pe <= 0:
            return {"可计算": False, "说明": "需要更多重复实验点"}

        ms_lof = ss_lof / dof_lof
        ms_pe = ss_pe / dof_pe
        f_lof = ms_lof / ms_pe if ms_pe > 0 else float("inf")
        p_lof = 1 - stats.f.cdf(f_lof, dof_lof, dof_pe)

        return {
            "SS_失拟": ss_lof, "SS_纯误差": ss_pe,
            "DOF_失拟": dof_lof, "DOF_纯误差": dof_pe,
            "F值": f_lof, "p值": p_lof,
            "显著失拟": p_lof < 0.05,
        }


def _pretty_name(raw: str, factor_names: list[str]) -> str:
    name = raw
    for i, fn in enumerate(factor_names):
        name = name.replace(f"x{i}", fn)
    return name.replace("^", "²").replace("1", "截距", 1) if name == "1" else name


def _term_metadata(raw: str, factor_names: list[str], column_index: int) -> dict:
    if " " in raw:
        factor_indices = tuple(int(part[1:]) for part in raw.split())
        label = " × ".join(factor_names[i] for i in factor_indices)
        return {
            "column_index": column_index,
            "factor_indices": factor_indices,
            "label": label,
            "type_label": "交互",
            "sort_group": 2,
        }

    if "^" in raw:
        base, _ = raw.split("^", maxsplit=1)
        factor_idx = int(base[1:])
        return {
            "column_index": column_index,
            "factor_indices": (factor_idx,),
            "label": f"{factor_names[factor_idx]}²",
            "type_label": "二次",
            "sort_group": 1,
        }

    factor_idx = int(raw[1:])
    return {
        "column_index": column_index,
        "factor_indices": (factor_idx,),
        "label": factor_names[factor_idx],
        "type_label": "线性",
        "sort_group": 0,
    }


def _ols_sse(X_design: np.ndarray, y: np.ndarray) -> float:
    coef = np.linalg.lstsq(X_design, y, rcond=None)[0]
    residuals = y - X_design @ coef
    return float(np.sum(residuals ** 2))
