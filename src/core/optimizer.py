"""护具位置优化器 - 自动计算最优佩戴位置

使用预计算和向量化方法加速优化搜索：
1. 预计算足部点云 KD-tree 和法线
2. 使用最近点法向法替代射线投射（结果一致，速度更快）
3. 向量化批量计算，避免 Python 循环
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import time

import numpy as np
import vtk

from .radial_distance_calculator import RadialDistanceCalculator, AxisCalculator


@dataclass
class OptimizationResult:
    """优化结果"""
    translation: np.ndarray       # 最优平移向量 (x, y, z)
    rotation: np.ndarray          # 最优旋转向量 (rx, ry, rz)
    ideal_ratio: float            # 理想区间 (4-6mm) 顶点比例
    ideal_count: int              # 理想区间顶点数
    total_count: int              # 总计算顶点数
    mean_distance: float          # 平均距离
    std_distance: float           # 距离标准差
    coverage_4_6mm: float         # 4-6mm 覆盖率


class BraceOptimizer:
    """
    护具位置优化器

    目标：找到护具相对足部的最优位置，使得空腔内侧表面与足部距离
    保持在 4-6mm 的顶点比例最大化

    使用多尺度搜索策略：
    1. 粗搜索：大范围、大步长
    2. 精搜索：小范围、小步长

    使用径向距离计算（基于中轴线），避免内外判断问题

    优化方法：
    - 预计算足部点云 KD-tree 和法线
    - 向量化批量计算距离
    - 进度提示和预计时间显示
    """

    def __init__(
        self,
        foot_polydata: vtk.vtkPolyData,
        brace_polydata: vtk.vtkPolyData,
        inner_surface_indices: np.ndarray,
    ):
        """
        参数:
            foot_polydata: 足部模型
            brace_polydata: 护具模型
            inner_surface_indices: 空腔内侧表面顶点索引
        """
        self.foot_polydata = foot_polydata
        self.brace_polydata = brace_polydata
        self.inner_surface_indices = inner_surface_indices

        # 预计算足部点云和法线（用于快速距离计算）
        self.foot_points, self.foot_normals = self._extract_points_with_normals(
            foot_polydata
        )

        # 构建 KD-tree 用于快速最近点查询
        from scipy.spatial import cKDTree
        self.foot_tree = cKDTree(self.foot_points)

        # 使用 PCA 拟合中轴线
        self.axis_origin, self.axis_direction = AxisCalculator.fit_axis_pca(
            brace_polydata, inner_surface_indices
        )

        # 提取内侧面顶点用于变换
        self.inner_vertices = self._extract_points_by_indices(
            brace_polydata, inner_surface_indices
        )

        # 计算内侧面顶点的局部坐标（相对于轴线）
        self._compute_vertex_local_coords()

        # 计算足部包围盒，用于确定搜索范围
        self.foot_bounds = self._compute_bounds(self.foot_points)
        self.brace_bounds = self._compute_bounds(self.inner_vertices)

        # 进度回调函数
        self.progress_callback: Optional[Callable] = None

    @staticmethod
    def _extract_points_with_normals(
        polydata: vtk.vtkPolyData
    ) -> Tuple[np.ndarray, np.ndarray]:
        """提取 vtkPolyData 的顶点数组和法线"""
        # 确保有法线
        if polydata.GetPointData().GetNormals() is None:
            normals_filter = vtk.vtkPolyDataNormals()
            normals_filter.SetInputData(polydata)
            normals_filter.ComputePointNormalsOn()
            normals_filter.ComputeCellNormalsOn()
            normals_filter.ConsistencyOn()
            normals_filter.Update()
            polydata = normals_filter.GetOutput()

        points = polydata.GetPoints()
        n = points.GetNumberOfPoints()
        point_arr = np.zeros((n, 3), dtype=np.float64)
        normal_arr = np.zeros((n, 3), dtype=np.float64)

        for i in range(n):
            point_arr[i] = points.GetPoint(i)
            normal_arr[i] = polydata.GetPointData().GetNormals().GetTuple(i)

        return point_arr, normal_arr

    @staticmethod
    def _extract_points(polydata: vtk.vtkPolyData) -> np.ndarray:
        """提取 vtkPolyData 的顶点数组"""
        points = polydata.GetPoints()
        n = points.GetNumberOfPoints()
        arr = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            arr[i] = points.GetPoint(i)
        return arr

    @staticmethod
    def _extract_points_by_indices(
        polydata: vtk.vtkPolyData,
        indices: np.ndarray
    ) -> np.ndarray:
        """按索引提取顶点"""
        points = polydata.GetPoints()
        n = len(indices)
        arr = np.zeros((n, 3), dtype=np.float64)
        for i, idx in enumerate(indices):
            arr[i] = points.GetPoint(idx)
        return arr

    def _compute_vertex_local_coords(self):
        """计算内侧面顶点相对于轴线的局部坐标"""
        # 计算每个顶点在轴线上的投影点和垂直距离
        v = self.inner_vertices - self.axis_origin
        proj_lengths = np.dot(v, self.axis_direction)  # 投影长度
        proj_points = self.axis_origin + np.outer(proj_lengths, self.axis_direction)

        # 顶点到轴线的垂直向量
        perp_vectors = self.inner_vertices - proj_points

        # 垂直距离（即到轴线的径向距离）
        self.vertex_radial_distances = np.linalg.norm(perp_vectors, axis=1)

        # 存储投影点（用于快速变换）
        self.vertex_proj_points = proj_points

    @staticmethod
    def _compute_bounds(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算点云的包围盒"""
        return points.min(axis=0), points.max(axis=0)

    def _apply_transform(
        self,
        translation: np.ndarray,
        rotation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        对内侧面顶点应用变换

        参数:
            translation: 平移向量 (3,)
            rotation: 旋转角度 (3,)，如果为 None 则不旋转

        返回:
            变换后的顶点 (N, 3)
        """
        if rotation is None:
            return self.inner_vertices + translation
        else:
            # 绕质心旋转
            centroid = self.inner_vertices.mean(axis=0)
            R = self._rotation_matrix_from_euler(rotation)
            return (
                np.dot(self.inner_vertices - centroid, R.T) + centroid + translation
            )

    def _compute_radial_distances_fast(self, transformed_vertices: np.ndarray) -> np.ndarray:
        """
        快速计算径向距离（向量化版本）

        原理：
        1. 使用 KD-tree 找到每个护具顶点在足部表面的最近点
        2. 计算护具顶点到轴线的垂直距离
        3. 计算足部最近点到轴线的垂直距离
        4. 径向间隙 = d_brace - d_foot

        参数:
            transformed_vertices: 变换后的护具内侧面顶点 (N, 3)

        返回:
            径向距离数组 (N,)
        """
        n = len(transformed_vertices)

        # 1. 使用 KD-tree 找到足部表面最近点（向量化）
        distances, indices = self.foot_tree.query(transformed_vertices, k=1)

        # 2. 获取最近点坐标
        foot_points = self.foot_points[indices]

        # 3. 计算护具顶点到轴线的垂直距离（向量化）
        v_brace = transformed_vertices - self.axis_origin
        proj_brace = np.dot(v_brace, self.axis_direction)
        closest_brace = self.axis_origin + np.outer(proj_brace, self.axis_direction)
        perp_brace = transformed_vertices - closest_brace
        d_brace = np.linalg.norm(perp_brace, axis=1)

        # 4. 计算足部最近点到轴线的垂直距离（向量化）
        v_foot = foot_points - self.axis_origin
        proj_foot = np.dot(v_foot, self.axis_direction)
        closest_foot = self.axis_origin + np.outer(proj_foot, self.axis_direction)
        perp_foot = foot_points - closest_foot
        d_foot = np.linalg.norm(perp_foot, axis=1)

        # 5. 径向间隙
        radial_gaps = d_brace - d_foot

        return radial_gaps

    def _evaluate_position(
        self,
        translation: np.ndarray,
        rotation: Optional[np.ndarray] = None
    ) -> float:
        """
        评估给定位置的优劣（使用快速径向距离计算）

        评分标准：
        1. 4-6mm 区间顶点比例（主要指标）
        2. 径向距离标准差（次要指标，越小越好）

        返回:
            综合得分（越高越好）
        """
        # 应用变换
        transformed = self._apply_transform(translation, rotation)

        # 快速计算径向距离（向量化）
        radial_gaps = self._compute_radial_distances_fast(transformed)

        # 统计
        total = len(radial_gaps)
        valid_mask = radial_gaps > 0  # 只统计有效值（有交点的）
        valid_count = np.sum(valid_mask)

        if valid_count == 0:
            return -np.inf

        # 4-6mm 区间比例
        ideal = np.sum(
            (radial_gaps[valid_mask] >= 4) & (radial_gaps[valid_mask] <= 6)
        )
        ideal_ratio = ideal / valid_count

        # 标准差惩罚
        std = np.std(radial_gaps[valid_mask])
        std_penalty = min(std / 10, 0.3)

        # 综合评分
        score = ideal_ratio - std_penalty

        return score

    @staticmethod
    def _rotation_matrix_from_euler(euler: np.ndarray) -> np.ndarray:
        """从欧拉角 (度) 计算旋转矩阵"""
        rx, ry, rz = np.radians(euler)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])

        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx

    def optimize(
        self,
        current_translation: np.ndarray = None,
        current_rotation: np.ndarray = None,
        search_range: float = 10.0,
        search_step: float = 2.0,
        rotation_range: float = 5.0,
        rotation_step: float = 2.5,
        fine_search_range: float = 2.0,
        fine_search_step: float = 0.5,
        fine_rotation_range: float = 2.5,
        fine_rotation_step: float = 0.5,
    ) -> OptimizationResult:
        """
        执行两级优化搜索（粗搜索 + 精搜索）

        参数:
            current_translation: 当前平移位置 (3,)，如果为 None 则使用质心对齐
            current_rotation: 当前旋转角度 (3,)，如果为 None 则使用零旋转
            search_range: 粗搜索平移范围半径 (mm)，默认±10mm
            search_step: 粗搜索平移步长 (mm)，默认 2mm
            rotation_range: 粗搜索旋转范围 (度)，默认±5 度
            rotation_step: 粗搜索旋转步长 (度)，默认 2.5 度
            fine_search_range: 精搜索平移范围半径 (mm)，默认±2mm
            fine_search_step: 精搜索平移步长 (mm)，默认 0.5mm
            fine_rotation_range: 精搜索旋转范围 (度)，默认±2.5 度
            fine_rotation_step: 精搜索旋转步长 (度)，默认 0.5 度

        返回:
            OptimizationResult 优化结果
        """
        # 初始位置：使用当前位置（如果没有则使用质心对齐）
        if current_translation is not None:
            initial_translation = current_translation.copy()
        else:
            foot_centroid = self.foot_points.mean(axis=0)
            brace_centroid = self.inner_vertices.mean(axis=0)
            initial_translation = foot_centroid - brace_centroid

        if current_rotation is not None:
            initial_rotation = current_rotation.copy()
        else:
            initial_rotation = np.zeros(3)

        # 计算搜索空间大小
        trans_offsets = np.arange(-search_range, search_range + 0.1, search_step)
        rot_offsets = np.arange(-rotation_range, rotation_range + 0.1, rotation_step)
        fine_trans_offsets = np.arange(-fine_search_range, fine_search_range + 0.1, fine_search_step)
        fine_rot_offsets = np.arange(-fine_rotation_range, fine_rotation_range + 0.1, fine_rotation_step)

        total_coarse_trans = len(trans_offsets) ** 3
        total_coarse_rot = len(rot_offsets) ** 3
        total_fine_trans = len(fine_trans_offsets) ** 3
        total_fine_rot = len(fine_rot_offsets) ** 3
        total_evaluations = total_coarse_trans + total_coarse_rot + total_fine_trans + total_fine_rot

        # 预估时间（基于每次评估约 15ms）
        estimated_time_seconds = total_evaluations * 0.015

        print("=" * 60)
        print(" 自动最优位置计算（使用快速径向距离算法）")
        print("=" * 60)
        print(f"\n【搜索空间】")
        print(f"  粗搜索平移：±{search_range}mm / {search_step}mm 步长 = {total_coarse_trans} 个位置")
        print(f"  粗搜索旋转：±{rotation_range}° / {rotation_step}° 步长 = {total_coarse_rot} 个组合")
        print(f"  精搜索平移：±{fine_search_range}mm / {fine_search_step}mm 步长 = {total_fine_trans} 个位置")
        print(f"  精搜索旋转：±{fine_rotation_range}° / {fine_rotation_step}° 步长 = {total_fine_rot} 个组合")
        print(f"\n【计算量】")
        print(f"  总评估次数：{total_evaluations:,} 次")
        print(f"  内侧面顶点数：{len(self.inner_vertices):,} 点")
        print(f"  预计耗时：约 {estimated_time_seconds:.0f}-{estimated_time_seconds*1.5:.0f} 秒")
        print(f"\n开始计算...")
        print("-" * 60)

        # 记录开始时间
        start_time = time.time()
        eval_count = 0

        # ========== 第一阶段：粗搜索 ==========
        print("\n【阶段 1/4】粗平移搜索...")
        stage_start = time.time()

        best_score = -np.inf
        best_translation = initial_translation.copy()
        best_rotation = initial_rotation.copy()

        # 生成平移搜索网格
        for dx in trans_offsets:
            for dy in trans_offsets:
                for dz in trans_offsets:
                    eval_count += 1
                    translation = initial_translation + np.array([dx, dy, dz])
                    score = self._evaluate_position(translation, initial_rotation)

                    if score > best_score:
                        best_score = score
                        best_translation = translation

                    # 进度显示（每 10% 显示一次）
                    progress = eval_count / total_evaluations * 100
                    if eval_count % max(1, total_coarse_trans // 10) == 0:
                        elapsed = time.time() - stage_start
                        remaining = (total_coarse_trans - eval_count) * (elapsed / max(1, eval_count))
                        print(f"  进度：{eval_count}/{total_coarse_trans} ({progress:.1f}%) | 剩余：{remaining:.0f}秒")

        stage_elapsed = time.time() - stage_start
        print(f"  完成！耗时：{stage_elapsed:.1f}秒")

        # 粗旋转搜索
        print(f"\n【阶段 2/4】粗旋转搜索...")
        stage_start = time.time()

        for rx in rot_offsets:
            for ry in rot_offsets:
                for rz in rot_offsets:
                    eval_count += 1
                    rotation = np.array([rx, ry, rz])
                    score = self._evaluate_position(best_translation, rotation)

                    if score > best_score:
                        best_score = score
                        best_rotation = rotation

                    progress = eval_count / total_evaluations * 100
                    if eval_count % max(1, total_coarse_rot // 10) == 0:
                        elapsed = time.time() - stage_start
                        remaining = (total_coarse_rot - eval_count) * (elapsed / max(1, eval_count - (total_coarse_trans)))
                        print(f"  进度：{eval_count - total_coarse_trans}/{total_coarse_rot} ({progress:.1f}%) | 剩余：{remaining:.0f}秒")

        stage_elapsed = time.time() - stage_start
        print(f"  完成！耗时：{stage_elapsed:.1f}秒")

        print(f"\n粗搜索完成 | 最佳得分：{best_score:.3f}")
        print(f"  平移：X={best_translation[0]:+.1f}, Y={best_translation[1]:+.1f}, Z={best_translation[2]:+.1f} mm")
        print(f"  旋转：RX={best_rotation[0]:+.1f}°, RY={best_rotation[1]:+.1f}°, RZ={best_rotation[2]:+.1f}°")

        # ========== 第二阶段：精搜索 ==========
        print("\n" + "=" * 50)
        print("第二阶段：精搜索（范围±2mm, ±2.5°）")
        print("=" * 50)

        fine_score = best_score
        fine_translation = best_translation.copy()
        fine_rotation = best_rotation.copy()

        # ========== 第二阶段：精搜索 ==========
        print("\n" + "-" * 60)
        print("【阶段 3/4】精平移搜索...")
        stage_start = time.time()

        # 精平移搜索（在粗搜索结果附近±2mm 范围内）
        for dx in fine_trans_offsets:
            for dy in fine_trans_offsets:
                for dz in fine_trans_offsets:
                    eval_count += 1
                    translation = best_translation + np.array([dx, dy, dz])
                    score = self._evaluate_position(translation, best_rotation)

                    if score > fine_score:
                        fine_score = score
                        fine_translation = translation

                    progress = eval_count / total_evaluations * 100
                    if eval_count % max(1, total_fine_trans // 10) == 0:
                        elapsed = time.time() - stage_start
                        remaining = (total_fine_trans - eval_count + total_fine_rot) * (elapsed / max(1, eval_count - (total_coarse_trans + total_coarse_rot)))
                        print(f"  进度：{eval_count - total_coarse_trans - total_coarse_rot}/{total_fine_trans} ({progress:.1f}%) | 剩余：{remaining:.0f}秒")

        stage_elapsed = time.time() - stage_start
        print(f"  完成！耗时：{stage_elapsed:.1f}秒")

        # 精细旋转搜索
        print(f"\n【阶段 4/4】精旋转搜索...")
        stage_start = time.time()

        for rx in fine_rot_offsets:
            for ry in fine_rot_offsets:
                for rz in fine_rot_offsets:
                    eval_count += 1
                    rotation = np.array([rx, ry, rz])
                    score = self._evaluate_position(fine_translation, rotation)

                    if score > fine_score:
                        fine_score = score
                        fine_rotation = rotation

                    progress = eval_count / total_evaluations * 100
                    if eval_count % max(1, total_fine_rot // 10) == 0:
                        elapsed = time.time() - stage_start
                        remaining = (total_fine_rot - (eval_count - (total_coarse_trans + total_coarse_rot + total_fine_trans))) * (elapsed / max(1, eval_count - (total_coarse_trans + total_coarse_rot + total_fine_trans)))
                        print(f"  进度：{eval_count - total_coarse_trans - total_coarse_rot - total_fine_trans}/{total_fine_rot} ({progress:.1f}%) | 剩余：{remaining:.0f}秒")

        stage_elapsed = time.time() - stage_start
        print(f"  完成！耗时：{stage_elapsed:.1f}秒")

        total_elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f" 优化完成 | 总耗时：{total_elapsed:.1f}秒")
        print(f"{'='*60}")
        print(f"  平移：X={fine_translation[0]:+.2f}, Y={fine_translation[1]:+.2f}, Z={fine_translation[2]:+.2f} mm")
        print(f"  旋转：RX={fine_rotation[0]:+.2f}°, RY={fine_rotation[1]:+.2f}°, RZ={fine_rotation[2]:+.2f}°")

        # 最终评估（使用快速方法）
        transformed = self._apply_transform(fine_translation, fine_rotation)
        radial_gaps = self._compute_radial_distances_fast(transformed)

        valid_mask = radial_gaps > 0
        valid_count = np.sum(valid_mask)
        ideal = np.sum((radial_gaps[valid_mask] >= 4) & (radial_gaps[valid_mask] <= 6))
        ideal_ratio = ideal / valid_count * 100 if valid_count > 0 else 0

        print(f"\n【结果统计】")
        print(f"  平均径向间隙：{np.mean(radial_gaps[valid_mask]):.2f} mm")
        print(f"  间隙标准差：{np.std(radial_gaps[valid_mask]):.2f} mm")
        print(f"  理想区间 (4-6mm): {ideal}/{valid_count} 点 ({ideal_ratio:.1f}%)")

        return OptimizationResult(
            translation=fine_translation,
            rotation=fine_rotation,
            ideal_ratio=ideal_ratio / 100,
            ideal_count=int(ideal),
            total_count=int(valid_count),
            mean_distance=float(np.mean(radial_gaps[valid_mask])),
            std_distance=float(np.std(radial_gaps[valid_mask])),
            coverage_4_6mm=ideal_ratio,
        )

    def optimize_grid_search(
        self,
        search_ranges: dict,
        step_size: float = 1.0
    ) -> OptimizationResult:
        """
        在指定范围内进行网格搜索

        参数:
            search_ranges: 搜索范围字典
                {'x': (min, max), 'y': (min, max), 'z': (min, max)}
            step_size: 搜索步长 (mm)

        返回:
            OptimizationResult 优化结果
        """
        best_score = -np.inf
        best_translation = np.zeros(3)

        x_range = search_ranges.get('x', (-5, 5))
        y_range = search_ranges.get('y', (-5, 5))
        z_range = search_ranges.get('z', (-5, 5))

        x_offsets = np.arange(x_range[0], x_range[1] + 0.1, step_size)
        y_offsets = np.arange(y_range[0], y_range[1] + 0.1, step_size)
        z_offsets = np.arange(z_range[0], z_range[1] + 0.1, step_size)

        total_positions = len(x_offsets) * len(y_offsets) * len(z_offsets)
        print(f"网格搜索：{len(x_offsets)} x {len(y_offsets)} x {len(z_offsets)} = {total_positions} 个位置")
        print(f"预计耗时：约 {total_positions * 0.015:.0f}-{total_positions * 0.02:.0f}秒")

        foot_centroid = self.foot_points.mean(axis=0)
        brace_centroid = self.inner_vertices.mean(axis=0)
        initial_translation = foot_centroid - brace_centroid

        eval_count = 0
        start_time = time.time()

        for dx in x_offsets:
            for dy in y_offsets:
                for dz in z_offsets:
                    eval_count += 1
                    translation = initial_translation + np.array([dx, dy, dz])
                    score = self._evaluate_position(translation)

                    if score > best_score:
                        best_score = score
                        best_translation = translation

                    if eval_count % max(1, total_positions // 10) == 0:
                        elapsed = time.time() - start_time
                        remaining = (total_positions - eval_count) * (elapsed / max(1, eval_count))
                        print(f"  进度：{eval_count}/{total_positions} | 剩余：{remaining:.0f}秒")

        # 最终评估（使用快速方法）
        transformed = self._apply_transform(best_translation, None)
        radial_gaps = self._compute_radial_distances_fast(transformed)

        valid_mask = radial_gaps > 0
        valid_count = np.sum(valid_mask)
        ideal = np.sum((radial_gaps[valid_mask] >= 4) & (radial_gaps[valid_mask] <= 6))
        ideal_ratio = ideal / valid_count * 100 if valid_count > 0 else 0

        print(f"\n网格搜索完成 | 总耗时：{time.time() - start_time:.1f}秒")
        print(f"  理想区间 (4-6mm): {ideal}/{valid_count} 点 ({ideal_ratio:.1f}%)")

        return OptimizationResult(
            translation=best_translation,
            rotation=np.zeros(3),
            ideal_ratio=ideal_ratio / 100,
            ideal_count=int(ideal),
            total_count=int(valid_count),
            mean_distance=float(np.mean(radial_gaps[valid_mask])),
            std_distance=float(np.std(radial_gaps[valid_mask])),
            coverage_4_6mm=ideal_ratio,
        )
