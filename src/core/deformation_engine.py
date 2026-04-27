"""变形引擎 — 对护具网格顶点应用尺寸修改"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import vtk


@dataclass
class DeformationParams:
    """变形参数"""
    mode: str  # "adaptive", "radial", "normal", "directional"
    region_indices: np.ndarray  # 受影响的顶点索引
    offset_mm: float = 0.0  # 偏移量 (mm)，正=放大，负=缩小
    scale_factor: float = 1.0  # 缩放因子
    decay_radius: float = 0.0  # 衰减半径 (mm)，0 = 无衰减
    direction: np.ndarray = None  # 自定义拉伸方向 (仅 directional 模式)
    center_point: np.ndarray = None  # 变形中心点


class DeformationEngine:
    """
    对护具内表面顶点应用变形，返回变形后的顶点数组

    所有变形方法均基于网格顶点操作，支持平滑衰减过渡。
    变形参数可记录并在 STEP 层面重新应用。
    """

    def __init__(
        self,
        foot_polydata: vtk.vtkPolyData,
        brace_polydata: vtk.vtkPolyData,
        inner_indices: np.ndarray,
    ):
        self.foot_polydata = foot_polydata
        self.brace_polydata = brace_polydata
        self.inner_indices = inner_indices

        # 构建足部 CellLocator 用于找最近点
        self._foot_locator = vtk.vtkCellLocator()
        self._foot_locator.SetDataSet(foot_polydata)
        self._foot_locator.BuildLocator()

        # 计算护具内表面法线
        self._normals = self._compute_normals(brace_polydata, inner_indices)

    @staticmethod
    def _compute_normals(
        polydata: vtk.vtkPolyData,
        indices: np.ndarray,
    ) -> np.ndarray:
        """计算指定顶点的法线向量

        使用网格质心校正法线方向，确保内表面法线一致朝向足部（向内）。
        """
        pts = polydata.GetPoints()
        n = len(indices)

        # 使用 vtkPolyDataNormals 计算原始法线
        norm_filter = vtk.vtkPolyDataNormals()
        norm_filter.SetInputData(polydata)
        norm_filter.ComputePointNormalsOn()
        norm_filter.ComputeCellNormalsOff()
        norm_filter.ConsistencyOn()
        norm_filter.SplittingOff()
        norm_filter.Update()

        vtk_normals = norm_filter.GetOutput().GetPointData().GetNormals()

        # 用质心校正法线方向：内表面法线应朝向质心（向内）
        centroid = np.zeros(3, dtype=np.float64)
        for i in range(n):
            idx = int(indices[i])
            centroid += np.array(pts.GetPoint(idx))
        centroid /= n

        normals = np.zeros((n, 3), dtype=np.float64)
        for i, idx in enumerate(indices):
            idx = int(idx)
            if vtk_normals:
                raw_normal = np.array(vtk_normals.GetTuple(idx))
            else:
                raw_normal = np.array(pts.GetPoint(idx)) - centroid

            # 校正法线方向：确保朝向质心（向内）
            to_centroid = centroid - np.array(pts.GetPoint(idx))
            if np.dot(raw_normal, to_centroid) < 0:
                raw_normal = -raw_normal  # 翻转，使法线朝向质心

            normals[i] = raw_normal

        # 归一化
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        normals /= norms
        return normals

    def _compute_weights(
        self,
        vertices: np.ndarray,
        center: np.ndarray,
        decay_radius: float,
    ) -> np.ndarray:
        """
        计算空间衰减权重

        使用二次衰减: w(d) = max(0, 1 - d/decay_radius)^2
        """
        if decay_radius <= 0:
            return np.ones(len(vertices), dtype=np.float64)

        distances = np.linalg.norm(vertices - center, axis=1)
        ratio = distances / decay_radius
        weights = np.maximum(0.0, 1.0 - ratio) ** 2
        return weights

    def _find_closest_foot_points(
        self, brace_vertices: np.ndarray
    ) -> np.ndarray:
        """查找每个护具顶点在足部表面上的最近点"""
        n = len(brace_vertices)
        foot_points = np.zeros((n, 3), dtype=np.float64)
        closest_point = np.zeros(3, dtype=np.float64)
        cell_id = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        dist2 = vtk.mutable(0.0)

        for i in range(n):
            self._foot_locator.FindClosestPoint(
                brace_vertices[i], closest_point, cell_id, sub_id, dist2
            )
            foot_points[i] = closest_point.copy()

        return foot_points

    def apply_adaptive(
        self,
        vertices: np.ndarray,
        target_gap: float = 5.0,
        current_gaps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        自适应模式：自动调整护具尺寸使平均间隙接近 target_gap

        原理：
        1. 计算当前平均间隙
        2. 沿内表面法线方向均匀偏移所有顶点
        3. 偏移量 = target_gap - current_mean_gap

        参数:
            vertices: 护具内表面顶点 (N, 3)
            target_gap: 目标间隙 (mm)
            current_gaps: 当前逐顶点间隙 (N,)，None 时自动计算

        返回:
            变形后的顶点数组 (N, 3)
        """
        if current_gaps is None:
            foot_points = self._find_closest_foot_points(vertices)
            current_gaps = np.linalg.norm(vertices - foot_points, axis=1)

        mean_gap = np.mean(current_gaps)
        offset = target_gap - mean_gap

        return self.apply_normal(
            vertices,
            DeformationParams(
                mode="normal",
                region_indices=np.arange(len(vertices)),
                offset_mm=offset,
            ),
        )

    def apply_radial(
        self,
        vertices: np.ndarray,
        params: DeformationParams,
    ) -> np.ndarray:
        """
        径向模式：沿中轴线径向缩放

        原理：
        1. 拟合中轴线（PCA 主轴）
        2. 对选区内顶点：沿垂直于轴线的方向缩放
        3. 带衰减：远离变形中心的点受影响递减
        """
        result = vertices.copy()

        # 拟合中轴线
        axis_origin, axis_direction = self._fit_mid_axis(vertices)

        if params.center_point is None:
            center = np.mean(vertices, axis=0)
        else:
            center = params.center_point

        # 选区顶点
        indices = params.region_indices
        selected = vertices[indices]

        # 计算每个选区顶点到轴线的投影
        v = selected - axis_origin
        proj = np.dot(v, axis_direction)[:, np.newaxis] * axis_direction
        radial_vec = selected - axis_origin - proj  # 径向向量

        # 计算权重（衰减）
        weights = self._compute_weights(
            selected, center, params.decay_radius
        )

        # 应用径向缩放
        scale_delta = params.scale_factor - 1.0
        for i in range(len(indices)):
            w = weights[i]
            result[indices[i]] += scale_delta * radial_vec[i] * w

        return result

    def apply_normal(
        self,
        vertices: np.ndarray,
        params: DeformationParams,
    ) -> np.ndarray:
        """
        法向模式：沿顶点法线方向偏移

        v' = v + offset * normal(v) * weight(distance)
        """
        result = vertices.copy()
        indices = params.region_indices
        selected = vertices[indices]
        selected_normals = self._normals[indices]

        if params.center_point is None:
            center = np.mean(vertices, axis=0)
        else:
            center = params.center_point

        weights = self._compute_weights(
            selected, center, params.decay_radius
        )

        for i in range(len(indices)):
            w = weights[i]
            result[indices[i]] += (
                params.offset_mm * selected_normals[i] * w
            )

        return result

    def apply_directional(
        self,
        vertices: np.ndarray,
        params: DeformationParams,
    ) -> np.ndarray:
        """
        方向拉伸：选区顶点沿指定方向拉伸

        v' = v + offset * direction * weight(distance)
        """
        result = vertices.copy()
        indices = params.region_indices
        selected = vertices[indices]

        direction = params.direction
        if direction is None:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm

        if params.center_point is None:
            center = np.mean(selected, axis=0)
        else:
            center = params.center_point

        weights = self._compute_weights(
            selected, center, params.decay_radius
        )

        for i in range(len(indices)):
            w = weights[i]
            result[indices[i]] += (
                params.offset_mm * direction * w
            )

        return result

    def apply(
        self,
        vertices: np.ndarray,
        params: DeformationParams,
    ) -> np.ndarray:
        """
        统一入口：根据模式调用对应的变形方法

        参数:
            vertices: 全部护具内表面顶点 (N, 3)
            params: 变形参数

        返回:
            变形后的顶点数组 (N, 3)
        """
        if params.mode == "adaptive":
            return self.apply_adaptive(vertices)
        elif params.mode == "radial":
            return self.apply_radial(vertices, params)
        elif params.mode == "normal":
            return self.apply_normal(vertices, params)
        elif params.mode == "directional":
            return self.apply_directional(vertices, params)
        else:
            raise ValueError(f"未知变形模式: {params.mode}")

    @staticmethod
    def _fit_mid_axis(
        vertices: np.ndarray,
    ) -> tuple:
        """
        使用 PCA 拟合顶点集的中轴线

        返回:
            (origin, direction) — 轴线上一点和单位方向向量
        """
        centroid = np.mean(vertices, axis=0)
        centered = vertices - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        main_axis = eigenvectors[:, np.argmax(eigenvalues)]

        # 确保主轴朝 Z 正方向
        if main_axis[2] < 0:
            main_axis = -main_axis

        return centroid, main_axis
