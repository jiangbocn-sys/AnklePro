"""变形引擎 — 对护具网格顶点应用尺寸修改

核心思路：
内表面是一个连续的弹性曲面。变形时不应让每个顶点沿自己的法线
独立移动（法线方向各异导致边缘撕裂），而应使用统一的位移方向场，
配合空间衰减权重实现平滑过渡。

类比：用手指按橡胶薄膜——薄膜整体鼓起，相邻区域平滑过渡，不会出现折痕。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import vtk


@dataclass
class DeformationParams:
    """变形参数"""
    mode: str  # "normal", "radial", "adaptive", "directional"
    region_indices: np.ndarray  # 受影响的顶点索引
    offset_mm: float = 0.0  # 偏移量 (mm)，正=放大，负=缩小
    scale_factor: float = 1.0  # 缩放因子
    decay_radius: float = 0.0  # 衰减半径 (mm)，0 = 无衰减
    boundary_smooth: float = 0.0  # 边界平滑半径 (mm)，0 = 不平滑
    direction: np.ndarray = None  # 自定义拉伸方向 (仅 directional 模式)
    center_point: np.ndarray = None  # 变形中心点
    _target_gap: float = 5.0  # 自适应模式目标间隙 (mm)


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

        # 缓存内表面顶点坐标和位移方向场
        self._inner_vertices = self._extract_vertices(brace_polydata, inner_indices)
        self._directions = self._compute_direction_field(self._inner_vertices)

        # 预计算内表面顶点法线（用于 directional 模式）
        self._normals = self._compute_surface_normals()

        # 构建内表面顶点 locator 用于找最近点
        self._inner_locator = self._build_locator(self._inner_vertices)

        # 缓存所有护具顶点（用于外壳联动）
        self._all_vertices = self._extract_vertices(brace_polydata, np.arange(brace_polydata.GetNumberOfPoints()))
        self._inner_set = set(int(idx) for idx in inner_indices)

    def _extract_vertices(
        self, polydata: vtk.vtkPolyData, indices: np.ndarray
    ) -> np.ndarray:
        """提取 polydata 中指定索引的顶点坐标"""
        pts = polydata.GetPoints()
        n = len(indices)
        vertices = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            vertices[i] = pts.GetPoint(int(indices[i]))
        return vertices

    def _compute_direction_field(
        self, vertices: np.ndarray
    ) -> np.ndarray:
        """
        计算统一位移方向场

        物理类比：内表面是一个弹性曲面（类似气球表面）。
        "向外膨胀" = 每个顶点沿从曲面质心指向自己的方向移动
        "向内收缩" = 反向

        这样保证：
        1. 所有顶点的位移方向统一且连续
        2. 曲面变形后整体鼓起/收缩，不会出现边缘折痕
        3. 对于护具这种近似圆柱/球面的形状，径向方向 ≈ 表面法线
           但方向场的一致性远优于逐顶点法线
        """
        centroid = np.mean(vertices, axis=0)
        directions = vertices - centroid  # 从质心指向各顶点

        # 归一化
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        directions /= norms

        return directions

    def _compute_surface_normals(self) -> np.ndarray:
        """
        预计算内表面各顶点的表面法线

        使用 vtkPolyDataNormals 计算网格法线，然后在 _inner_vertices 中查找对应法线。
        """
        from vtk.util.numpy_support import vtk_to_numpy

        # 提取内表面的 polydata 以便计算法线
        inner_poly = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        for v in self._inner_vertices:
            pts.InsertNextPoint(v[0], v[1], v[2])
        inner_poly.SetPoints(pts)

        # 复制内表面的 cell 结构
        cell_array = self.brace_polydata.GetPolys()
        cell_array.InitTraversal()
        id_list = vtk.vtkIdList()
        idx_set = set(int(idx) for idx in self.inner_indices)
        idx_to_inner = {int(idx): i for i, idx in enumerate(self.inner_indices)}

        new_cells = vtk.vtkCellArray()
        while cell_array.GetNextCell(id_list):
            cell_ids = []
            for k in range(id_list.GetNumberOfIds()):
                cid = id_list.GetId(k)
                if cid in idx_set:
                    cell_ids.append(idx_to_inner[cid])
            if len(cell_ids) >= 3:
                new_cells.InsertNextCell(len(cell_ids), cell_ids)
        inner_poly.SetPolys(new_cells)

        # 计算法线
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(inner_poly)
        normals_filter.ComputePointNormalsOn()
        normals_filter.ComputeCellNormalsOff()
        normals_filter.AutoOrientNormalsOn()
        normals_filter.Update()

        normals_data = normals_filter.GetOutput().GetPointData().GetNormals()
        if normals_data:
            return vtk_to_numpy(normals_data)
        return np.zeros_like(self._inner_vertices)

    @staticmethod
    def _build_locator(vertices: np.ndarray) -> vtk.vtkPointLocator:
        """
        对顶点集构建 PointLocator，用于快速查找最近顶点
        """
        pts = vtk.vtkPoints()
        for v in vertices:
            pts.InsertNextPoint(v[0], v[1], v[2])
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)

        locator = vtk.vtkPointLocator()
        locator.SetDataSet(poly)
        locator.BuildLocator()
        return locator

    def _find_nearest_inner_vertex(self, point: np.ndarray) -> int:
        """
        在 _inner_vertices 中找到距离 point 最近的顶点索引（数组索引，0..N-1）
        """
        return int(self._inner_locator.FindClosestPoint(point))

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

    def apply_normal(
        self,
        vertices: np.ndarray,
        params: DeformationParams,
    ) -> np.ndarray:
        """
        法向模式：变形空腔形状，外壳跟随，壁厚恒定。

        与 directional 模式完全一致：对**所有顶点**计算平滑权重场，
        每个顶点都有自己的位移（非最近邻继承），外壳无缝联动。
        """
        n = len(vertices)
        result = vertices.copy()
        inner_idx_map = {int(idx): i for i, idx in enumerate(self.inner_indices)}

        # ---- 变形中心点 ----
        if params.center_point is not None:
            center = params.center_point
        else:
            center = np.mean(self._inner_vertices, axis=0)

        # ---- 动态衰减半径 ----
        inner_extent = np.ptp(self._inner_vertices, axis=0).max()
        if params.decay_radius > 0:
            effective_radius = max(params.decay_radius, inner_extent * 0.8)
        else:
            effective_radius = inner_extent * 1.2

        # ---- 所有顶点统一平滑权重场（基于到中心的距离） ----
        distances = np.linalg.norm(vertices - center, axis=1)
        ratio = distances / effective_radius
        all_weights = np.maximum(0.0, np.cos(np.pi * np.minimum(ratio, 1.0) / 2.0)) ** 2

        # ---- 方向场扩展到所有顶点 ----
        # 内表面顶点：预计算的方向场
        # 外壳顶点：通过最近邻内表面顶点的方向
        from scipy.spatial import cKDTree
        tree = cKDTree(self._inner_vertices)
        _, nn_indices = tree.query(vertices, k=1)
        all_directions = self._directions[nn_indices]

        # ---- 应用位移 ----
        for i in range(n):
            if all_weights[i] > 0:
                result[i] += params.offset_mm * all_directions[i] * all_weights[i]

        return result

    def apply_normal_z(
        self,
        vertices: np.ndarray,
        params: DeformationParams,
    ) -> np.ndarray:
        """
        法向 Z轴圆柱模式：沿 Z 轴做圆柱形膨胀/收缩。

        与质心放射模式的区别：
        - 质心放射：每个顶点沿质心→自己的方向移动 → 球形/弧形膨胀
        - Z轴圆柱：每个顶点沿垂直于Z轴的方向（径向远离Z轴）移动 → 圆柱形膨胀

        方向场计算：
        1. 取内表面顶点在 Z 轴上的中心线（XY 平面上的投影质心）
        2. 每个顶点沿垂直于 Z 轴的方向（XY 平面内的径向）位移
        3. 权重基于到中心线的距离（圆柱半径）
        """
        n = len(vertices)
        result = vertices.copy()

        # ---- Z轴中心线：内表面顶点在XY平面上的投影质心 ----
        xy_center = np.mean(self._inner_vertices[:, :2], axis=0)  # (x_center, y_center)
        center_3d = np.array([xy_center[0], xy_center[1], np.mean(self._inner_vertices[:, 2])])

        # ---- 动态衰减半径 ----
        inner_extent = np.ptp(self._inner_vertices, axis=0).max()
        if params.decay_radius > 0:
            effective_radius = max(params.decay_radius, inner_extent * 0.8)
        else:
            effective_radius = inner_extent * 1.2

        # ---- 所有顶点统一平滑权重场 ----
        distances = np.linalg.norm(vertices - center_3d, axis=1)
        ratio = distances / effective_radius
        all_weights = np.maximum(0.0, np.cos(np.pi * np.minimum(ratio, 1.0) / 2.0)) ** 2

        # ---- Z轴圆柱方向场：垂直于Z轴的径向方向 ----
        radial_xy = vertices - center_3d
        radial_xy[:, 2] = 0  # 只保留 XY 分量
        norms = np.linalg.norm(radial_xy, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        radial_xy /= norms  # 归一化为 XY 平面内的单位方向

        # ---- 应用位移 ----
        for i in range(n):
            if all_weights[i] > 0:
                result[i] += params.offset_mm * radial_xy[i] * all_weights[i]

        return result

    def _smooth_weights(
        self,
        weights: np.ndarray,
        region_indices: np.ndarray,
        smooth_radius: float,
    ) -> np.ndarray:
        """
        对权重标量场做拉普拉斯平滑

        1. 区域顶点权重始终保持 >= 1.0
        2. 非区域顶点从邻居传播获得权重
        3. 多次迭代产生平滑过渡带
        """
        n = len(weights)
        result = weights.copy()

        # 构建内表面邻接关系
        cell_array = self.brace_polydata.GetPolys()
        cell_array.InitTraversal()
        id_list = vtk.vtkIdList()
        idx_to_array = {int(idx): i for i, idx in enumerate(self.inner_indices)}

        adjacency = {i: [] for i in range(n)}
        while cell_array.GetNextCell(id_list):
            cell_ids = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]
            array_ids = []
            for cid in cell_ids:
                if cid in idx_to_array:
                    array_ids.append(idx_to_array[cid])
            for i, aid in enumerate(array_ids):
                for j, other in enumerate(array_ids):
                    if i != j and other not in adjacency[aid]:
                        adjacency[aid].append(other)

        is_in_region = np.zeros(n, dtype=bool)
        is_in_region[region_indices] = True

        iterations = max(1, min(15, int(smooth_radius / 1.5)))

        for _ in range(iterations):
            new_weights = result.copy()
            for i in range(n):
                if is_in_region[i]:
                    continue  # 区域顶点权重不变
                if not adjacency[i]:
                    continue

                neighbor_sum = 0.0
                count = 0
                for j in adjacency[i]:
                    if result[j] > 0:
                        neighbor_sum += result[j]
                        count += 1
                if count > 0:
                    new_weights[i] = neighbor_sum / count
            result = new_weights

        return result

    def apply_adaptive(
        self,
        vertices: np.ndarray,
        target_gap: Optional[float] = None,
        current_gaps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        自适应模式：自动调整护具尺寸使平均间隙接近 target_gap
        """
        if target_gap is None:
            target_gap = 5.0
        if current_gaps is None:
            foot_points = self._find_closest_foot_points(self._inner_vertices)
            current_gaps = np.linalg.norm(self._inner_vertices - foot_points, axis=1)

        mean_gap = np.mean(current_gaps)
        offset = target_gap - mean_gap

        return self.apply_normal(
            vertices,
            DeformationParams(
                mode="normal",
                region_indices=np.arange(len(self._inner_vertices)),
                offset_mm=offset,
            ),
        )

    def apply_radial(
        self,
        vertices: np.ndarray,
        params: DeformationParams,
    ) -> np.ndarray:
        """
        径向模式：沿中轴线径向缩放空腔，外壳跟随，壁厚恒定。

        与 directional/normal 模式完全一致：对**所有顶点**计算平滑权重场，
        每个顶点直接获得自己的位移。
        """
        from scipy.spatial import cKDTree

        n = len(vertices)
        result = vertices.copy()

        axis_origin, axis_direction = self._get_radial_axis(
            self._inner_vertices, params.direction
        )

        # ---- 变形中心点 ----
        if params.center_point is not None:
            center = params.center_point
        else:
            center = np.mean(self._inner_vertices, axis=0)

        # ---- 动态衰减半径 ----
        inner_extent = np.ptp(self._inner_vertices, axis=0).max()
        if params.decay_radius > 0:
            effective_radius = max(params.decay_radius, inner_extent * 0.8)
        else:
            effective_radius = inner_extent * 1.2

        # ---- 所有顶点统一平滑权重场（基于到中心的距离） ----
        distances = np.linalg.norm(vertices - center, axis=1)
        ratio = distances / effective_radius
        all_weights = np.maximum(0.0, np.cos(np.pi * np.minimum(ratio, 1.0) / 2.0)) ** 2

        # ---- 计算每个顶点相对于中轴线的径向向量 ----
        v = vertices - axis_origin
        proj = np.dot(v, axis_direction)[:, np.newaxis] * axis_direction
        radial_vec = v - proj  # (N, 3)

        # ---- 应用位移 ----
        scale_delta = params.scale_factor - 1.0
        active = all_weights > 0
        result[active] += scale_delta * radial_vec[active] * all_weights[active, np.newaxis]

        return result

    def compute_directional_field(
        self,
        params: DeformationParams,
    ) -> tuple:
        """
        计算方向变形的完整位移场（用于外壳联动）

        返回: (center, direction, weights) — 可用于对任意顶点集计算位移
        """
        inner_to_all = {int(idx): i for i, idx in enumerate(self.inner_indices)}

        # 确定变形中心点
        if params.center_point is not None:
            center = params.center_point
        else:
            foot_points = self._find_closest_foot_points(
                self._inner_vertices[params.region_indices]
            )
            gaps = np.linalg.norm(
                self._inner_vertices[params.region_indices] - foot_points, axis=1
            )
            region_closest_idx = int(np.argmin(gaps))
            closest_idx = int(params.region_indices[region_closest_idx])
            center = self._inner_vertices[closest_idx]

        # 确定位移方向
        if params.direction is not None:
            norm = np.linalg.norm(params.direction)
            direction = params.direction / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])
        else:
            nearest_idx = self._find_nearest_inner_vertex(center)
            direction = self._normals[nearest_idx]
            # 确保方向远离足部（向外 = 扩大空腔）
            foot_pt = self._find_closest_foot_points(
                self._inner_vertices[nearest_idx:nearest_idx+1]
            )[0]
            to_foot = foot_pt - self._inner_vertices[nearest_idx]
            if np.dot(direction, to_foot) > 0:
                direction = -direction
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-10:
                direction = direction / direction_norm

        # ---- 动态衰减半径 ----
        # 如果用户指定的半径太小（小于模型尺寸的80%），自动放大
        inner_extent = np.ptp(self._inner_vertices, axis=0).max()
        if params.decay_radius > 0:
            effective_radius = max(params.decay_radius, inner_extent * 0.8)
        else:
            effective_radius = inner_extent * 1.2

        # ---- 所有顶点统一权重 ----
        # 护具是一个整体，变形影响所有在衰减半径内的顶点。
        # 权重仅由顶点到选点的距离决定。
        distances = np.linalg.norm(self._all_vertices - center, axis=1)
        ratio = distances / effective_radius
        final_weights = np.maximum(0.0, np.cos(np.pi * np.minimum(ratio, 1.0) / 2.0)) ** 2

        return center, direction, final_weights

    def apply_directional(
        self,
        vertices: np.ndarray,
        params: DeformationParams,
    ) -> np.ndarray:
        """
        点驱动变形（手指按弹性塑料板）

        内侧面顶点：按距离衰减权重变形
        外侧面顶点：直接继承内侧面位移向量（壁厚恒定）
        """
        from scipy.spatial import cKDTree

        center, direction, weights = self.compute_directional_field(params)

        result = vertices.copy()
        inner_idx_map = {int(idx): i for i, idx in enumerate(self.inner_indices)}

        # ---- 内表面变形 ----
        inner_disp = np.zeros((len(self._inner_vertices), 3), dtype=np.float64)
        for inner_array_idx in range(len(self._inner_vertices)):
            mesh_idx = int(self.inner_indices[inner_array_idx])
            d = params.offset_mm * direction * weights[mesh_idx]
            result[mesh_idx] += d
            inner_disp[inner_array_idx] = d

        # ---- 外壳联动：直接继承内表面位移向量 ----
        outer_indices = np.array([i for i in range(len(vertices)) if i not in inner_idx_map])
        if len(outer_indices) > 0:
            tree = cKDTree(self._inner_vertices)
            _, nn_indices = tree.query(vertices[outer_indices], k=1)
            result[outer_indices] += inner_disp[nn_indices]

        return result

    def apply(
        self,
        vertices: np.ndarray,
        params: DeformationParams,
    ) -> np.ndarray:
        """统一入口：根据模式调用对应的变形方法"""
        if params.mode == "adaptive":
            return self.apply_adaptive(vertices, params._target_gap)
        elif params.mode == "radial":
            return self.apply_radial(vertices, params)
        elif params.mode == "normal":
            return self.apply_normal(vertices, params)
        elif params.mode == "normal_z":
            return self.apply_normal_z(vertices, params)
        elif params.mode == "directional":
            return self.apply_directional(vertices, params)
        else:
            raise ValueError(f"未知变形模式: {params.mode}")

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

    @staticmethod
    def _fit_mid_axis(
        vertices: np.ndarray,
    ) -> tuple:
        """使用 PCA 拟合顶点集的中轴线"""
        centroid = np.mean(vertices, axis=0)
        centered = vertices - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        main_axis = eigenvectors[:, np.argmax(eigenvalues)]

        if main_axis[2] < 0:
            main_axis = -main_axis

        return centroid, main_axis

    @staticmethod
    def _get_radial_axis(
        vertices: np.ndarray,
        user_axis: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        获取径向轴

        如果用户指定了轴向，使用用户选择（原点在顶点质心）
        否则使用 PCA 拟合中轴线
        """
        centroid = np.mean(vertices, axis=0)

        if user_axis is not None:
            norm = np.linalg.norm(user_axis)
            if norm > 1e-10:
                user_axis = user_axis / norm
            return centroid, user_axis

        return DeformationEngine._fit_mid_axis(vertices)
