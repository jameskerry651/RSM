"""优化模块 — 在响应面上寻找极值."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, differential_evolution

from rsm.model import RSMModel


@dataclass
class OptimizationResult:
    optimal_coded: np.ndarray
    optimal_natural: np.ndarray
    predicted_response: float
    factor_names: list[str]
    success: bool

    def summary(self) -> dict:
        result = {}
        for name, coded, natural in zip(self.factor_names, self.optimal_coded, self.optimal_natural):
            result[name] = {"编码值": round(coded, 4), "自然值": round(natural, 4)}
        result["预测响应"] = round(self.predicted_response, 4)
        return result


class RSMOptimizer:

    def __init__(self, model: RSMModel, factors):
        self.model = model
        self.factors = factors

    def optimize(
        self,
        maximize: bool = True,
        coded_bounds: tuple[float, float] = (-1.5, 1.5),
        method: str = "differential_evolution",
    ) -> OptimizationResult:
        k = len(self.factors)
        bounds = [coded_bounds] * k
        sign = -1 if maximize else 1

        def objective(x):
            return sign * self.model.predict(x.reshape(1, -1))[0]

        if method == "differential_evolution":
            result = differential_evolution(objective, bounds, seed=42, maxiter=500, tol=1e-10)
        else:
            x0 = np.zeros(k)
            result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

        optimal_coded = result.x
        optimal_natural = np.array([f.decode(optimal_coded[i]) for i, f in enumerate(self.factors)])
        predicted = self.model.predict(optimal_coded.reshape(1, -1))[0]

        return OptimizationResult(
            optimal_coded=optimal_coded,
            optimal_natural=optimal_natural,
            predicted_response=predicted,
            factor_names=[f.name for f in self.factors],
            success=result.success,
        )

    def canonical_analysis(self) -> dict:
        """典型分析: 计算驻点与特征值, 判定响应面形态."""
        if self.model.order != 2:
            return {"错误": "典型分析需要二阶模型"}

        k = len(self.factors)
        coefs = self.model.coefficients
        # 从系数中提取 b (线性), B (二次矩阵)
        # 系数排列: [截距, x1, x2, ..., x1^2, x1*x2, ..., x2^2, ...]
        b = coefs[1:k + 1]
        B = np.zeros((k, k))

        idx = k + 1
        for i in range(k):
            B[i, i] = coefs[idx]
            idx += 1
            for j in range(i + 1, k):
                B[i, j] = coefs[idx] / 2
                B[j, i] = coefs[idx] / 2
                idx += 1

        eigenvalues, eigenvectors = np.linalg.eigh(B)

        try:
            stationary_coded = -0.5 * np.linalg.solve(B, b)
        except np.linalg.LinAlgError:
            stationary_coded = np.zeros(k)

        stationary_natural = np.array([f.decode(stationary_coded[i]) for i, f in enumerate(self.factors)])
        predicted_at_sp = self.model.predict(stationary_coded.reshape(1, -1))[0]

        if np.all(eigenvalues < 0):
            shape = "极大值 (Maximum)"
        elif np.all(eigenvalues > 0):
            shape = "极小值 (Minimum)"
        else:
            shape = "鞍点 (Saddle Point)"

        return {
            "驻点(编码)": stationary_coded,
            "驻点(自然)": stationary_natural,
            "驻点预测值": predicted_at_sp,
            "特征值": eigenvalues,
            "特征向量": eigenvectors,
            "曲面形态": shape,
        }
