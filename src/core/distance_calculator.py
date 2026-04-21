"""距离计算器 — 使用 vtkOBBTree 加速的射线投射法计算有符号距离"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import vtk


@dataclass
class DistanceStats:
    """距离统计信息"""
    distances: np.ndarray     # 逐顶点有符号距离 (mm)
    min_dist: float
    max_dist: float
    mean_dist: float
    median_dist: float
    std_dist: float
    penetration_count: int    # 穿透点数 (距离 < 0)
    ideal_count: int          # 理想区间点数


class DistanceCalculator:
    """
    计算护具顶点到足部表面的有符号距离

    使用 vtkOBBTree 进行空间加速：
    - FindClosestPoint: 找到最近点
    - LineIntersection: 快速射线相交测试（OBB 树加速）

    内外判断：射线投射法（奇内偶外原理），不受法线方向影响
    """

    def __init__(self, foot_polydata: vtk.vtkPolyData):
        """
        参数:
            foot_polydata: 足部模型的 vtkPolyData
        """
        self._foot_polydata = self._ensure_normals(foot_polydata)

        # CellLocator 用于找最近点
        self._locator = vtk.vtkCellLocator()
        self._locator.SetDataSet(self._foot_polydata)
        self._locator.BuildLocator()

        # OBBTree 用于快速射线相交测试
        self._obb_tree = vtk.vtkOBBTree()
        self._obb_tree.SetDataSet(self._foot_polydata)
        self._obb_tree.BuildLocator()

    @staticmethod
    def _ensure_normals(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """确保 polydata 有面法线"""
        if polydata.GetPointData().GetNormals() is None:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(polydata)
            normals.ComputePointNormalsOn()
            normals.ComputeCellNormalsOn()
            normals.ConsistencyOn()
            normals.Update()
            return normals.GetOutput()
        return polydata

    def _is_inside(self, point: np.ndarray) -> bool:
        """
        判断点是否在足部封闭曲面内部

        使用射线投射法 + OBBTree 加速：
        - 奇数个交点 = 内部
        - 偶数个交点 = 外部

        参数:
            point: 待检查的 3D 点坐标

        返回:
            True = 内部，False = 外部
        """
        # 向 X 轴正方向发射射线
        ray_start = point.copy()
        ray_end = point + np.array([1000.0, 0.0, 0.0])  # 足够远

        # 使用 OBBTree 的 LineIntersection 快速计算交点
        intersections = vtk.vtkPoints()
        self._obb_tree.IntersectWithLine(
            ray_start, ray_end, intersections, None
        )

        intersection_count = intersections.GetNumberOfPoints()
        return (intersection_count % 2) == 1

    def compute_signed_distances(
        self, brace_vertices: np.ndarray
    ) -> np.ndarray:
        """
        计算护具顶点到足部表面的有符号距离

        对每个护具顶点：
        1. 用 CellLocator.FindClosestPoint 找到最近点
        2. 计算欧氏距离
        3. 用射线投射法判断内外（OBBTree 加速）

        参数:
            brace_vertices: 护具内侧面顶点 (N, 3)

        返回:
            有符号距离数组 (N,)，单位 mm
        """
        n = len(brace_vertices)
        distances = np.zeros(n, dtype=np.float64)
        closest_point = np.zeros(3, dtype=np.float64)

        cell_id = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        dist2 = vtk.mutable(0.0)

        for i in range(n):
            pt = brace_vertices[i]

            # 找最近点
            self._locator.FindClosestPoint(
                pt, closest_point, cell_id, sub_id, dist2
            )
            dist = np.sqrt(float(dist2))

            # 射线投射法判断内外
            if self._is_inside(pt):
                distances[i] = -dist
            else:
                distances[i] = dist

        return distances

    def compute_with_stats(
        self, brace_vertices: np.ndarray,
        thresholds: Optional[list] = None
    ) -> DistanceStats:
        """计算距离并返回统计信息"""
        distances = self.compute_signed_distances(brace_vertices)

        ideal_min = 4.0
        ideal_max = 6.0
        if thresholds:
            for t_min, t_max, color, label in thresholds:
                if "理想" in label or "ideal" in label.lower():
                    ideal_min = t_min if t_min != float("-inf") else 0.0
                    ideal_max = t_max if t_max != float("inf") else 999.0
                    break

        ideal_count = int(
            np.sum((distances >= ideal_min) & (distances <= ideal_max))
        )

        return DistanceStats(
            distances=distances,
            min_dist=float(np.min(distances)),
            max_dist=float(np.max(distances)),
            mean_dist=float(np.mean(distances)),
            median_dist=float(np.median(distances)),
            std_dist=float(np.std(distances)),
            penetration_count=int(np.sum(distances < 0)),
            ideal_count=ideal_count,
        )
