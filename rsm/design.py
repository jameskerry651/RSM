"""实验设计模块 — 生成 CCD、Box-Behnken、全因子等设计矩阵."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pyDOE3 import bbdesign, ccdesign


@dataclass
class Factor:
    name: str
    low: float
    high: float
    unit: str = ""

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2

    @property
    def half_range(self) -> float:
        return (self.high - self.low) / 2

    def encode(self, natural: float) -> float:
        return (natural - self.center) / self.half_range

    def decode(self, coded: float) -> float:
        return coded * self.half_range + self.center


@dataclass
class ExperimentDesigner:
    factors: list[Factor] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.factors)

    @property
    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    # ---- design generators ----

    def full_factorial(self, levels: int = 3) -> pd.DataFrame:
        vals = np.linspace(-1, 1, levels)
        grid = np.array(list(itertools.product(vals, repeat=self.k)))
        return self._to_dataframe(grid)

    def central_composite(self, alpha: str = "orthogonal", face: str = "circumscribed") -> pd.DataFrame:
        coded = ccdesign(self.k, alpha=alpha, face=face)
        return self._to_dataframe(coded)

    def box_behnken(self) -> pd.DataFrame:
        if self.k < 3:
            raise ValueError("Box-Behnken 设计至少需要 3 个因子")
        coded = bbdesign(self.k)
        return self._to_dataframe(coded)

    def custom(self, coded_matrix: np.ndarray) -> pd.DataFrame:
        return self._to_dataframe(coded_matrix)

    # ---- helpers ----

    def _to_dataframe(self, coded: np.ndarray) -> pd.DataFrame:
        natural = np.column_stack(
            [f.decode(coded[:, i]) for i, f in enumerate(self.factors)]
        )
        df = pd.DataFrame(natural, columns=self.factor_names)
        for i, f in enumerate(self.factors):
            df[f"coded_{f.name}"] = coded[:, i]
        return df

    def encode_df(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [self.factors[i].encode(df[f.name].values) for i, f in enumerate(self.factors)]
        )

    def summary(self) -> pd.DataFrame:
        rows = []
        for f in self.factors:
            rows.append({"因子": f.name, "低水平": f.low, "高水平": f.high, "中心点": f.center, "单位": f.unit})
        return pd.DataFrame(rows)
