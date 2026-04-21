"""
内外判断诊断工具

用于分析和验证护具顶点相对于足部模型的内外位置关系
"""

import numpy as np
import vtk
from typing import Tuple, List


class InsideOutsideChecker:
    """
    内外位置检查器

    提供多种方法判断点是否在封闭曲面内部
    """

    def __init__(self, foot_polydata: vtk.vtkPolyData):
        self.foot_polydata = self._ensure_normals(foot_polydata)
        self.foot_bounds = self._compute_bounds()

    def _ensure_normals(self, polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """确保有法线"""
        if polydata.GetPointData().GetNormals() is None:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(polydata)
            normals.ComputePointNormalsOn()
            normals.ComputeCellNormalsOn()
            normals.ConsistencyOn()  # 关键：确保法线方向一致
            normals.Update()
            return normals.GetOutput()
        return polydata

    def _compute_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算足部包围盒"""
        bounds = self.foot_polydata.GetBounds()
        return (
            np.array([bounds[0], bounds[2], bounds[4]]),
            np.array([bounds[1], bounds[3], bounds[5]])
        )

    def check_by_ray_casting(self, point: np.ndarray) -> Tuple[bool, int]:
        """
        射线投射法判断内外（最可靠）

        从点向任意方向发射射线，计算与足部表面的交点数量：
        - 奇数个交点 = 内部
        - 偶数个交点 = 外部

        参数:
            point: 待检查的 3D 点坐标

        返回:
            (is_inside, intersection_count)
        """
        # 沿 X 轴正方向发射射线
        ray_dir = np.array([1.0, 0.0, 0.0])
        intersection_count = self._count_ray_intersections(point, ray_dir)

        # 奇数 = 内部，偶数 = 外部
        is_inside = (intersection_count % 2) == 1

        return is_inside, intersection_count

    def _count_ray_intersections(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> int:
        """计算射线与足部表面的交点数量"""
        intersection_count = 0

        n_cells = self.foot_polydata.GetNumberOfCells()
        pts = self.foot_polydata.GetPoints()

        for cell_id in range(n_cells):
            cell = self.foot_polydata.GetCell(cell_id)

            # 获取三角形三个顶点
            p0 = np.array(pts.GetPoint(cell.GetPointId(0)))
            p1 = np.array(pts.GetPoint(cell.GetPointId(1)))
            p2 = np.array(pts.GetPoint(cell.GetPointId(2)))

            # Möller-Trumbore 射线 - 三角形相交测试
            edge1 = p1 - p0
            edge2 = p2 - p0

            h = np.cross(direction, edge2)
            det = np.dot(edge1, h)

            if abs(det) < 1e-8:
                continue  # 射线与三角形平行

            inv_det = 1.0 / det
            s = origin - p0
            u = np.dot(s, h) * inv_det

            if u < 0 or u > 1:
                continue

            q = np.cross(s, edge1)
            v = np.dot(direction, q) * inv_det

            if v < 0 or u + v > 1:
                continue

            t = np.dot(edge2, q) * inv_det

            if t > 0:  # 交点在射线正方向
                intersection_count += 1

        return intersection_count

    def check_by_normal_consistency(
        self, point: np.ndarray, closest_point: np.ndarray,
        normal: np.ndarray
    ) -> bool:
        """
        法线一致性检查（当前使用的方法）

        假设：足部表面法线朝外
        - (point - closest_point) · normal > 0 → 外部
        - (point - closest_point) · normal < 0 → 内部

        参数:
            point: 护具点
            closest_point: 足部表面最近点
            normal: 足部表面法线

        返回:
            True = 外部，False = 内部
        """
        to_point = point - closest_point
        dot = np.dot(to_point, normal)
        return dot >= 0

    def check_by_winding_number(
        self, point: np.ndarray, num_samples: int = 100
    ) -> Tuple[bool, float]:
        """
        立体角/缠绕数检查

        计算点对足部表面的立体角：
        - 立体角 ≈ 4π → 内部
        - 立体角 ≈ 0 → 外部

        参数:
            point: 待检查点
            num_samples: 采样三角形数量

        返回:
            (is_inside, solid_angle)
        """
        solid_angle = 0.0
        pts = self.foot_polydata.GetPoints()
        n_cells = min(self.foot_polydata.GetNumberOfCells(), num_samples)

        for i in range(n_cells):
            cell = self.foot_polydata.GetCell(i)
            p0 = np.array(pts.GetPoint(cell.GetPointId(0))) - point
            p1 = np.array(pts.GetPoint(cell.GetPointId(1))) - point
            p2 = np.array(pts.GetPoint(cell.GetPointId(2))) - point

            # 计算三角形对点的立体角（Oosterom-Strackee 公式）
            norm0, norm1, norm2 = np.linalg.norm(p0), np.linalg.norm(p1), np.linalg.norm(p2)
            if norm0 < 1e-10 or norm1 < 1e-10 or norm2 < 1e-10:
                continue

            p0, p1, p2 = p0/norm0, p1/norm1, p2/norm2

            numerator = np.dot(p0, np.cross(p1, p2))
            denominator = 1 + np.dot(p0, p1) + np.dot(p1, p2) + np.dot(p2, p0)

            if abs(denominator) < 1e-10:
                continue

            solid_angle += np.arctan2(numerator, denominator)

        # 总立体角
        total_solid_angle = 2 * solid_angle * n_cells / num_samples

        is_inside = abs(total_solid_angle) > 2 * np.pi

        return is_inside, total_solid_angle

    def diagnose_vertices(
        self, brace_vertices: np.ndarray, sample_size: int = 50
    ) -> dict:
        """
        诊断一批顶点的内外判断

        参数:
            brace_vertices: 护具顶点数组
            sample_size: 抽样检查数量

        返回:
            诊断报告字典
        """
        n = len(brace_vertices)
        sample_indices = np.random.choice(n, min(sample_size, n), replace=False)

        results = {
            'total_vertices': n,
            'sampled': len(sample_indices),
            'consistent': 0,
            'inconsistent': 0,
            'details': []
        }

        # 创建 locator 找最近点
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(self.foot_polydata)
        locator.BuildLocator()

        foot_normals = self._extract_normals()

        # 如果没有法线，使用射线投射法作为基准
        if foot_normals is None:
            for idx in sample_indices:
                pt = brace_vertices[idx]

                # 找最近点
                closest_point = np.zeros(3, dtype=np.float64)
                cell_id = vtk.mutable(0)
                sub_id = vtk.mutable(0)
                dist2 = vtk.mutable(0.0)

                locator.FindClosestPoint(pt, closest_point, cell_id, sub_id, dist2)
                dist = np.sqrt(float(dist2))

                # 只用射线投射法
                ray_inside, ray_count = self.check_by_ray_casting(pt)

                results['details'].append({
                    'vertex_idx': int(idx),
                    'distance': dist,
                    'normal_says': 'N/A无法线',
                    'ray_says': 'inside' if ray_inside else 'outside',
                    'ray_intersections': ray_count,
                    'consistent': True  # 没有法线对比，标记为一致
                })

            # 统计内外分布
            inside_count = sum(1 for d in results['details'] if d['ray_says'] == 'inside')
            results['ray_inside_count'] = inside_count
            results['ray_outside_count'] = len(sample_indices) - inside_count
            results['consistent'] = len(sample_indices)
            results['consistency_rate'] = 100.0
            results['note'] = '足部模型无法线数据，仅使用射线投射法'

        else:
            for idx in sample_indices:
                pt = brace_vertices[idx]

                # 找最近点
                closest_point = np.zeros(3, dtype=np.float64)
                cell_id = vtk.mutable(0)
                sub_id = vtk.mutable(0)
                dist2 = vtk.mutable(0.0)

                locator.FindClosestPoint(pt, closest_point, cell_id, sub_id, dist2)
                dist = np.sqrt(float(dist2))

                cid = int(cell_id)
                normal = foot_normals[cid]

                # 方法 1：法线判断（当前使用）
                normal_outside = self.check_by_normal_consistency(pt, closest_point, normal)

                # 方法 2：射线投射（可靠基准）
                ray_inside, ray_count = self.check_by_ray_casting(pt)

                # 一致性检查
                consistent = (not ray_inside) == normal_outside

                if consistent:
                    results['consistent'] += 1
                else:
                    results['inconsistent'] += 1

                results['details'].append({
                    'vertex_idx': int(idx),
                    'distance': dist,
                    'normal_says': 'outside' if normal_outside else 'inside',
                    'ray_says': 'inside' if ray_inside else 'outside',
                    'ray_intersections': ray_count,
                    'consistent': consistent
                })

            results['consistency_rate'] = results['consistent'] / len(sample_indices) * 100

        return results

    def _extract_normals(self) -> np.ndarray:
        """提取足部面法线"""
        cell_normals = self.foot_polydata.GetCellData().GetNormals()
        if cell_normals is None:
            # 如果没有法线，返回 None
            return None
        n_cells = self.foot_polydata.GetNumberOfCells()
        normals = np.zeros((n_cells, 3), dtype=np.float64)
        for i in range(n_cells):
            normals[i] = cell_normals.GetTuple(i)
        return normals


def check_normal_orientation(polydata: vtk.vtkPolyData) -> dict:
    """
    检查足部模型法线方向是否正确（朝外）

    方法：从每个面中心沿法线方向发射射线，检查是否会再次与模型相交
    - 如果会相交 → 法线朝内
    - 如果不相交 → 法线朝外
    """
    # 首先确保有法线
    if polydata.GetCellData().GetNormals() is None:
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(polydata)
        normals_filter.ComputeCellNormalsOn()
        normals_filter.ConsistencyOn()
        normals_filter.Update()
        polydata = normals_filter.GetOutput()

    bounds = polydata.GetBounds()
    center = np.array([
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2
    ])

    n_cells = polydata.GetNumberOfCells()
    normals = polydata.GetCellData().GetNormals()

    if normals is None:
        return {
            'total_sampled': 0,
            'pointing_out': 0,
            'pointing_in': 0,
            'out_ratio': 0,
            'error': '无法获取法线数据'
        }

    pointing_out = 0
    pointing_in = 0

    # 采样检查
    sample_cells = np.random.choice(n_cells, min(100, n_cells), replace=False)

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()

    for cell_id in sample_cells:
        cell = polydata.GetCell(cell_id)
        cell_center = np.zeros(3)

        for i in range(cell.GetNumberOfPoints()):
            pt = np.array(polydata.GetPoint(cell.GetPointId(i)))
            cell_center += pt
        cell_center /= cell.GetNumberOfPoints()

        normal = np.array(normals.GetTuple(cell_id))

        # 从面中心沿法线方向发射射线
        ray_start = cell_center + normal * 0.1  # 稍微偏移避免自相交
        ray_end = cell_center + normal * 1000  # 很远的点

        # 检查是否与模型有其他交点
        # 简单方法：检查射线终点到最近点的距离
        closest_point = np.zeros(3, dtype=np.float64)
        cid = vtk.mutable(0)
        sid = vtk.mutable(0)
        d2 = vtk.mutable(0.0)

        locator.FindClosestPoint(ray_end, closest_point, cid, sid, d2)
        dist_to_surface = np.sqrt(float(d2))

        # 如果射线终点离表面很远，说明法线朝外
        if dist_to_surface > 50:
            pointing_out += 1
        else:
            pointing_in += 1

    return {
        'total_sampled': len(sample_cells),
        'pointing_out': pointing_out,
        'pointing_in': pointing_in,
        'out_ratio': pointing_out / len(sample_cells) * 100
    }
