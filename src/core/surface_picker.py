"""表面选取工具 — 识别护具空腔内侧面"""

from typing import Optional, Tuple

import numpy as np
import vtk


def identify_inner_surface(
    polydata: vtk.vtkPolyData,
    angle_threshold: float = 90.0
) -> np.ndarray:
    """
    自动识别护具空腔内侧面

    原理：内侧面的法线朝向护具中心（足部所在位置）。
    选取所有法线与"面中心→护具质心"方向夹角小于阈值的三角面。

    参数:
        polydata: 护具模型的 vtkPolyData
        angle_threshold: 最大夹角（度），默认90度

    返回:
        被选中的三角面（cell）索引数组
    """
    centroid = _compute_centroid(polydata)
    cell_normals = _get_cell_normals(polydata)
    cell_centers = _get_cell_centers(polydata)
    n_cells = polydata.GetNumberOfCells()
    to_centroid = centroid - cell_centers
    dot_products = np.sum(cell_normals * to_centroid, axis=1)
    norms_normals = np.linalg.norm(cell_normals, axis=1)
    norms_to_centroid = np.linalg.norm(to_centroid, axis=1)
    cos_angles = dot_products / (norms_normals * norms_to_centroid + 1e-10)
    inner_mask = cos_angles > np.cos(np.radians(angle_threshold))
    return np.where(inner_mask)[0]


def pick_cell(
    actor: vtk.vtkActor,
    polydata: vtk.vtkPolyData,
    renderer: vtk.vtkRenderer,
    display_x: int,
    display_y: int
) -> Optional[int]:
    """
    在屏幕上点击一个像素，拾取对应的三角面（cell）

    参数:
        actor: 护具的 vtk Actor
        polydata: 护具的 vtkPolyData
        renderer: VTK 渲染器
        display_x, display_y: 鼠标点击的屏幕坐标

    返回:
        cell ID，如果没点到则返回 None
    """
    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.001)
    picker.Pick(display_x, display_y, 0, renderer)

    if picker.GetActor() == actor:
        return picker.GetCellId()
    return None


def extract_inner_vertices(
    polydata: vtk.vtkPolyData,
    inner_cells: np.ndarray
) -> np.ndarray:
    """提取内侧面涉及的所有顶点索引"""
    vertex_set = set()
    for cell_id in inner_cells:
        cell = polydata.GetCell(cell_id)
        for i in range(cell.GetNumberOfPoints()):
            vertex_set.add(cell.GetPointId(i))
    return np.array(sorted(vertex_set), dtype=np.int64)


def find_connected_regions(
    inner_cells: np.ndarray,
    polydata: vtk.vtkPolyData,
    angle_threshold: float = 75.0,
    min_region_size: int = 100,
) -> list:
    """
    将内侧面集合拆分为多个连通区域

    使用优化的并查集数据结构加速连通性判断。

    参数:
        inner_cells: 内侧面 cell 索引数组
        polydata: 护具模型的 vtkPolyData
        angle_threshold: 法线夹角阈值（度），超过此角度视为断开
        min_region_size: 最小区域大小，小于此值的区域将被过滤

    返回:
        连通区域列表，每个区域是 cell 索引的 numpy 数组
    """
    if len(inner_cells) == 0:
        return []

    cell_normals = _get_cell_normals(polydata)
    inner_set = set(inner_cells.tolist())
    n_inner = len(inner_cells)

    # 使用并查集加速连通性合并
    parent = {cell_id: cell_id for cell_id in inner_set}

    def find(x):
        # 迭代版本的路径压缩，避免递归过深
        root = x
        while parent[root] != root:
            root = parent[root]
        # 路径压缩
        current = x
        while current != root:
            next_node = parent[current]
            parent[current] = root
            current = next_node
        return root

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # 构建点→cell 的映射，加速邻居查找
    point_to_cells = {}
    for cell_id in inner_set:
        cell = polydata.GetCell(cell_id)
        for j in range(cell.GetNumberOfPoints()):
            pid = cell.GetPointId(j)
            if pid not in point_to_cells:
                point_to_cells[pid] = []
            point_to_cells[pid].append(cell_id)

    # 通过共享边查找邻居并合并（比共享点更精确）
    edge_to_cell = {}
    for cell_id in inner_set:
        cell = polydata.GetCell(cell_id)
        n_pts = cell.GetNumberOfPoints()
        for i in range(n_pts):
            p1 = cell.GetPointId(i)
            p2 = cell.GetPointId((i + 1) % n_pts)
            # 使用排序的点对作为边的键
            edge_key = tuple(sorted([p1, p2]))
            if edge_key not in edge_to_cell:
                edge_to_cell[edge_key] = []
            edge_to_cell[edge_key].append(cell_id)

    # 通过共享边合并相邻面
    cos_threshold = np.cos(np.radians(angle_threshold))
    for edge_key, cells in edge_to_cell.items():
        if len(cells) < 2:
            continue
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                c1, c2 = cells[i], cells[j]
                # 检查法线夹角
                n1 = cell_normals[c1]
                n2 = cell_normals[c2]
                cos_angle = np.dot(n1, n2) / (
                    np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-10
                )
                if cos_angle >= cos_threshold:
                    union(c1, c2)

    # 按根节点分组
    region_map = {}
    for cell_id in inner_set:
        root = find(cell_id)
        if root not in region_map:
            region_map[root] = []
        region_map[root].append(cell_id)

    # 过滤小区域并转换为数组
    regions = []
    for cells in region_map.values():
        if len(cells) >= min_region_size:
            regions.append(np.array(cells, dtype=np.int64))

    # 按区域大小排序，大的在前
    regions.sort(key=len, reverse=True)

    return regions


def get_transformed_vertices(
    polydata: vtk.vtkPolyData,
    vertex_indices: np.ndarray,
    transform_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """获取指定顶点的坐标（可选择性地应用变换）"""
    pts = polydata.GetPoints()
    vertices = np.zeros((len(vertex_indices), 3), dtype=np.float64)
    for i, idx in enumerate(vertex_indices):
        vertices[i] = pts.GetPoint(idx)
    if transform_matrix is not None:
        hom = np.hstack([vertices, np.ones((len(vertices), 1))])
        vertices = (transform_matrix @ hom.T).T[:, :3]
    return vertices


# ---- 内部函数 ----

def _compute_centroid(polydata: vtk.vtkPolyData) -> np.ndarray:
    """计算模型几何中心"""
    pts = polydata.GetPoints()
    n = pts.GetNumberOfPoints()
    total = np.zeros(3, dtype=np.float64)
    for i in range(n):
        total += np.array(pts.GetPoint(i))
    return total / n


def _get_cell_normals(polydata: vtk.vtkPolyData) -> np.ndarray:
    """获取所有三角面的法向量 (M, 3)"""
    cell_normals = polydata.GetCellData().GetNormals()
    n_cells = polydata.GetNumberOfCells()
    normals = np.zeros((n_cells, 3), dtype=np.float64)
    for i in range(n_cells):
        normals[i] = cell_normals.GetTuple(i)
    return normals


def _get_cell_centers(polydata: vtk.vtkPolyData) -> np.ndarray:
    """获取所有三角面的中心坐标 (M, 3)"""
    n_cells = polydata.GetNumberOfCells()
    centers = np.zeros((n_cells, 3), dtype=np.float64)
    pts = polydata.GetPoints()
    for i in range(n_cells):
        cell = polydata.GetCell(i)
        center = np.zeros(3, dtype=np.float64)
        for j in range(cell.GetNumberOfPoints()):
            pt = np.array(pts.GetPoint(cell.GetPointId(j)))
            center += pt
        centers[i] = center / cell.GetNumberOfPoints()
    return centers


def _get_cell_neighbors(
    polydata: vtk.vtkPolyData,
    cell_id: int,
) -> list:
    """获取与指定cell共享顶点的邻居cell ID"""
    cell = vtk.vtkGenericCell()
    polydata.GetCell(cell_id, cell)
    neighbor_ids = vtk.vtkIdList()
    polydata.GetCellNeighbors(cell_id, cell.GetPointIds(), neighbor_ids)
    return [neighbor_ids.GetId(i) for i in range(neighbor_ids.GetNumberOfIds())]
