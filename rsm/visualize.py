"""可视化模块 — 3D 曲面、等高线图、残差诊断等."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from rsm.model import RSMModel


class RSMVisualizer:

    def __init__(self, model: RSMModel, factor_names: list[str] | None = None):
        self.model = model
        self.factor_names = factor_names or model._factor_names

    def surface_3d(
        self,
        x_idx: int = 0,
        y_idx: int = 1,
        hold_values: dict[int, float] | None = None,
        resolution: int = 50,
        title: str = "",
    ) -> go.Figure:
        grid_x = np.linspace(-1.5, 1.5, resolution)
        grid_y = np.linspace(-1.5, 1.5, resolution)
        xx, yy = np.meshgrid(grid_x, grid_y)

        k = len(self.factor_names)
        points = np.zeros((resolution * resolution, k))
        points[:, x_idx] = xx.ravel()
        points[:, y_idx] = yy.ravel()
        if hold_values:
            for idx, val in hold_values.items():
                points[:, idx] = val

        zz = self.model.predict(points).reshape(resolution, resolution)

        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=grid_x, y=grid_y, z=zz,
            colorscale="Viridis", opacity=0.9,
            contours_z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True),
        ))
        fig.update_layout(
            title=title or f"响应曲面: {self.factor_names[x_idx]} × {self.factor_names[y_idx]}",
            scene=dict(
                xaxis_title=self.factor_names[x_idx],
                yaxis_title=self.factor_names[y_idx],
                zaxis_title="响应值 (Y)",
            ),
            width=700, height=600,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig

    def contour(
        self,
        x_idx: int = 0,
        y_idx: int = 1,
        hold_values: dict[int, float] | None = None,
        resolution: int = 80,
        title: str = "",
    ) -> go.Figure:
        grid_x = np.linspace(-1.5, 1.5, resolution)
        grid_y = np.linspace(-1.5, 1.5, resolution)
        xx, yy = np.meshgrid(grid_x, grid_y)

        k = len(self.factor_names)
        points = np.zeros((resolution * resolution, k))
        points[:, x_idx] = xx.ravel()
        points[:, y_idx] = yy.ravel()
        if hold_values:
            for idx, val in hold_values.items():
                points[:, idx] = val

        zz = self.model.predict(points).reshape(resolution, resolution)

        fig = go.Figure()
        fig.add_trace(go.Contour(
            x=grid_x, y=grid_y, z=zz,
            colorscale="RdYlBu_r",
            contours=dict(showlabels=True, labelfont=dict(size=11)),
            colorbar=dict(title="响应值"),
        ))
        fig.update_layout(
            title=title or f"等高线图: {self.factor_names[x_idx]} × {self.factor_names[y_idx]}",
            xaxis_title=self.factor_names[x_idx],
            yaxis_title=self.factor_names[y_idx],
            width=650, height=550,
        )
        return fig

    def perturbation_plot(self, hold_coded: np.ndarray | None = None, n_points: int = 100) -> go.Figure:
        k = len(self.factor_names)
        if hold_coded is None:
            hold_coded = np.zeros(k)

        xs = np.linspace(-1.5, 1.5, n_points)
        fig = go.Figure()
        colors = ["#EF553B", "#636EFA", "#00CC96", "#AB63FA", "#FFA15A"]
        for i in range(k):
            points = np.tile(hold_coded, (n_points, 1))
            points[:, i] = xs
            ys = self.model.predict(points)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=self.factor_names[i],
                line=dict(width=2.5, color=colors[i % len(colors)]),
            ))

        fig.update_layout(
            title="扰动图 (Perturbation Plot)",
            xaxis_title="编码值",
            yaxis_title="响应值 (Y)",
            width=650, height=450,
            legend=dict(x=0.01, y=0.99),
        )
        return fig

    def residual_diagnostics(self) -> go.Figure:
        residuals = self.model.residuals
        fitted = self.model.fitted_values
        y = self.model._y
        std_res = (residuals - residuals.mean()) / (residuals.std() + 1e-12)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("残差 vs 拟合值", "正态概率图 (Q-Q)", "残差直方图", "实测 vs 预测"),
        )

        fig.add_trace(go.Scatter(
            x=fitted, y=residuals, mode="markers",
            marker=dict(size=8, color="#636EFA"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        qq = stats.probplot(std_res)
        fig.add_trace(go.Scatter(
            x=qq[0][0], y=qq[0][1], mode="markers",
            marker=dict(size=8, color="#EF553B"),
            showlegend=False,
        ), row=1, col=2)
        x_line = np.array([qq[0][0].min(), qq[0][0].max()])
        fig.add_trace(go.Scatter(
            x=x_line, y=qq[1][0] * x_line + qq[1][1],
            mode="lines", line=dict(dash="dash", color="gray"),
            showlegend=False,
        ), row=1, col=2)

        fig.add_trace(go.Histogram(
            x=residuals, nbinsx=max(8, len(residuals) // 3),
            marker_color="#00CC96",
            showlegend=False,
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=y, y=fitted, mode="markers",
            marker=dict(size=8, color="#AB63FA"),
            showlegend=False,
        ), row=2, col=2)
        xy_range = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
        fig.add_trace(go.Scatter(
            x=xy_range, y=xy_range, mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
        ), row=2, col=2)

        fig.update_layout(height=650, width=750, title_text="残差诊断", showlegend=False)
        fig.update_xaxes(title_text="拟合值", row=1, col=1)
        fig.update_yaxes(title_text="残差", row=1, col=1)
        fig.update_xaxes(title_text="理论分位数", row=1, col=2)
        fig.update_yaxes(title_text="标准化残差", row=1, col=2)
        fig.update_xaxes(title_text="残差", row=2, col=1)
        fig.update_yaxes(title_text="频数", row=2, col=1)
        fig.update_xaxes(title_text="实测值", row=2, col=2)
        fig.update_yaxes(title_text="预测值", row=2, col=2)
        return fig

    def interaction_plot(self, x_idx: int = 0, trace_idx: int = 1, levels: int = 5) -> go.Figure:
        k = len(self.factor_names)
        xs = np.linspace(-1.5, 1.5, 80)
        trace_vals = np.linspace(-1.5, 1.5, levels)
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

        fig = go.Figure()
        for j, tv in enumerate(trace_vals):
            points = np.zeros((len(xs), k))
            points[:, x_idx] = xs
            points[:, trace_idx] = tv
            ys = self.model.predict(points)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                name=f"{self.factor_names[trace_idx]}={tv:.1f}",
                line=dict(width=2.5, color=colors[j % len(colors)]),
            ))

        fig.update_layout(
            title=f"交互作用图: {self.factor_names[x_idx]} × {self.factor_names[trace_idx]}",
            xaxis_title=self.factor_names[x_idx],
            yaxis_title="响应值 (Y)",
            width=650, height=450,
        )
        return fig
