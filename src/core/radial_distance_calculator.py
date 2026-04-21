"""径向距离计算器 — 基于中轴线的护具 - 足部间隙测量"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import vtk


@dataclass
class RadialDistanceStats:
    """径向距离统计信息"""
    radial_distances: np.ndarray  # 径向间隙 (mm)
    d_brace: np.ndarray           # 护具顶点到轴线距离
    d_foot: np.ndarray            # 足部交点到轴线距离
    mean_gap: float
    std_gap: float
    min_gap: float
    max_gap: float
    ideal_4_6_ratio: float        # 4-6mm 区间比例
    ideal_4_6_count: int


class RadialDistanceCalculator:
    """
    基于中轴线的径向距离计算器

    原理：
    1. 确定护具空腔的中轴线（默认为 Z 轴，或通过 PCA 计算主轴）
    2. 对每个护具内表面顶点：
       - 向中轴线发射射线
       - 计算射线与足部表面的交点
       - 计算径向间隙 = d_brace - d_foot
    """

    def __init__(
        self,
        foot_polydata: vtk.vtkPolyData,
        brace_polydata: vtk.vtkPolyData,
        inner_vertex_indices: np.ndarray,
        axis_direction: np.ndarray = None,
        axis_origin: np.ndarray = None,
    ):
        """
        参数:
            foot_polydata: 足部模型
            brace_polydata: 护具模型
            inner_vertex_indices: 护具内表面顶点索引
            axis_direction: 中轴线方向（默认 Z 轴）
            axis_origin: 中轴线原点（默认护具质心）
        """
        self.foot_polydata = foot_polydata
        self.brace_polydata = brace_polydata
        self.inner_vertex_indices = inner_vertex_indices

        # 中轴线参数
        if axis_direction is None:
            self.axis_direction = np.array([0.0, 0.0, 1.0])  # Z 轴
        else:
            self.axis_direction = axis_direction / np.linalg.norm(axis_direction)

        # 计算护具内表面的质心作为轴线原点
        if axis_origin is None:
            self.axis_origin = self._compute_inner_centroid(
                brace_polydata, inner_vertex_indices
            )
        else:
            self.axis_origin = axis_origin

        # 构建足部 OBBTree 用于快速相交测试
        self.obb_tree = vtk.vtkOBBTree()
        self.obb_tree.SetDataSet(foot_polydata)
        self.obb_tree.BuildLocator()

    @staticmethod
    def _compute_inner_centroid(
        polydata: vtk.vtkPolyData,
        vertex_indices: np.ndarray
    ) -> np.ndarray:
        """计算内表面顶点的质心"""
        pts = polydata.GetPoints()
        n = len(vertex_indices)
        total = np.zeros(3)
        for idx in vertex_indices:
            total += np.array(pts.GetPoint(idx))
        return total / n

    def _point_to_line_distance(
        self, point: np.ndarray, line_origin: np.ndarray, line_dir: np.ndarray
    ) -> float:
        """
        计算点到直线的垂直距离

        参数:
            point: 3D 点坐标
            line_origin: 直线原点
            line_dir: 直线方向（单位向量）

        返回:
            垂直距离
        """
        # 从直线原点到点的向量
        v = point - line_origin

        # 投影到直线方向
        proj = np.dot(v, line_dir) * line_dir

        # 垂直分量
        perp = v - proj

        return np.linalg.norm(perp)

    def _ray_intersection(
        self, start: np.ndarray, direction: np.ndarray
    ) -> Tuple[bool, np.ndarray]:
        """
        计算射线与足部表面的交点

        参数:
            start: 射线起点
            direction: 射线方向（应指向轴线）

        返回:
            (有交点，交点坐标)
        """
        # 射线终点（足够远）
        ray_end = start + direction * 500

        intersections = vtk.vtkPoints()
        self.obb_tree.IntersectWithLine(
            start, ray_end, intersections, None
        )

        if intersections.GetNumberOfPoints() > 0:
            # 返回第一个交点（最近的）
            pt = intersections.GetPoint(0)
            return True, np.array(pt)
        else:
            return False, None

    def compute_radial_distances(
        self,
        brace_vertices: np.ndarray,
        transform_matrix: Optional[np.ndarray] = None
    ) -> RadialDistanceStats:
        """
        计算径向间隙

        参数:
            brace_vertices: 护具内表面顶点（已应用变换）
            transform_matrix: 护具的 4x4 变换矩阵（如果顶点未变换）

        返回:
            RadialDistanceStats 统计信息
        """
        n = len(brace_vertices)
        d_brace = np.zeros(n)  # 护具顶点到轴线距离
        d_foot = np.zeros(n)   # 足部交点到轴线距离
        radial_distances = np.zeros(n)

        valid_count = 0

        for i in range(n):
            pt_brace = brace_vertices[i]

            # 计算护具顶点到轴线的垂直距离
            d_b = self._point_to_line_distance(
                pt_brace, self.axis_origin, self.axis_direction
            )

            # 向轴线方向发射射线
            # 找到轴线上最近的点
            v = pt_brace - self.axis_origin
            proj = np.dot(v, self.axis_direction) * self.axis_direction
            closest_on_axis = self.axis_origin + proj

            # 从护具顶点指向轴线的方向
            to_axis = closest_on_axis - pt_brace
            if np.linalg.norm(to_axis) < 1e-6:
                continue
            to_axis = to_axis / np.linalg.norm(to_axis)

            # 计算与足部的交点
            has_intersection, pt_foot = self._ray_intersection(pt_brace, to_axis)

            if has_intersection:
                # 计算交点到轴线的距离
                d_f = self._point_to_line_distance(
                    pt_foot, self.axis_origin, self.axis_direction
                )

                # 径向间隙
                gap = d_b - d_f

                d_brace[i] = d_b
                d_foot[i] = d_f
                radial_distances[i] = gap
                valid_count += 1

        # 统计
        valid_mask = radial_distances != 0
        if np.sum(valid_mask) > 0:
            ideal_count = int(
                np.sum((radial_distances >= 4) & (radial_distances <= 6))
            )
            return RadialDistanceStats(
                radial_distances=radial_distances,
                d_brace=d_brace,
                d_foot=d_foot,
                mean_gap=float(np.mean(radial_distances[valid_mask])),
                std_gap=float(np.std(radial_distances[valid_mask])),
                min_gap=float(np.min(radial_distances[valid_mask])),
                max_gap=float(np.max(radial_distances[valid_mask])),
                ideal_4_6_ratio=float(ideal_count / np.sum(valid_mask) * 100),
                ideal_4_6_count=ideal_count,
            )
        else:
            return RadialDistanceStats(
                radial_distances=radial_distances,
                d_brace=d_brace,
                d_foot=d_foot,
                mean_gap=0,
                std_gap=0,
                min_gap=0,
                max_gap=0,
                ideal_4_6_ratio=0,
                ideal_4_6_count=0,
            )


class AxisCalculator:
    """
    中轴线计算器

    提供多种方法确定护具空腔的中轴线
    """

    @staticmethod
    def fit_axis_pca(polydata: vtk.vtkPolyData, vertex_indices: np.ndarray):
        """
        使用 PCA 拟合内表面顶点的主轴

        返回:
            (原点，方向)
        """
        pts = polydata.GetPoints()
        vertices = np.array([
            [pts.GetPoint(i)[j] for j in range(3)]
            for i in vertex_indices
        ])

        # 计算协方差矩阵
        centroid = vertices.mean(axis=0)
        centered = vertices - centroid
        cov = np.cov(centered.T)

        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 最大特征值对应的特征向量 = 主轴方向
        main_axis = eigenvectors[:, np.argmax(eigenvalues)]

        return centroid, main_axis

    @staticmethod
    def fit_cylinder_axis(polydata: vtk.vtkPolyData, vertex_indices: np.ndarray):
        """
        拟合圆柱面轴线（更精确但计算复杂）

        简单实现：使用 PCA 结果
        """
        return AxisCalculator.fit_axis_pca(polydata, vertex_indices)
