"""STL 文件结构对比 — 比较两个网格的拓扑和几何差异"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import vtk
from scipy.spatial import cKDTree

from src.core.model_loader import load_stl


@dataclass
class MeshStats:
    """网格基本统计信息"""
    vertex_count: int
    triangle_count: int
    bounds: tuple           # (min_x, min_y, min_z, max_x, max_y, max_z)
    extent: np.ndarray      # (x, y, z) 尺寸
    centroid: np.ndarray    # 质心
    volume: float           # 体积 (mm³)
    surface_area: float     # 表面积 (mm²)


@dataclass
class STLComparison:
    """两个 STL 文件的对比结果"""
    stats_a: MeshStats
    stats_b: MeshStats
    topology_match: bool            # 拓扑是否完全一致（相同顶点数+相同连接）
    topology_differences: list      # 拓扑差异描述
    vertex_distances: np.ndarray    # 每对最近顶点的距离 (N,)
    max_distance: float             # 最大顶点距离
    mean_distance: float            # 平均顶点距离
    rms_distance: float             # RMS 距离
    distance_percentiles: tuple     # (p10, p25, p50, p75, p90, p95, p99)
    overlap_ratio: float            # A 在 B 的公差范围内的顶点占比


def compare_stl(filepath_a: str, filepath_b: str,
                tolerance: float = 0.01,
                same_topology: bool = False) -> STLComparison:
    """
    对比两个 STL 文件的结构

    Args:
        filepath_a: 第一个 STL 文件路径
        filepath_b: 第二个 STL 文件路径
        tolerance: 重叠判定公差 (mm) — 距离小于此值认为顶点"重合"
        same_topology: 两个文件是否来自相同网格（顶点数和连接关系相同）
                       True 时做精确的顶点级对比，False 时做最近邻距离分析
    """
    model_a = load_stl(filepath_a)
    model_b = load_stl(filepath_b)

    stats_a = _compute_stats(model_a.polydata)
    stats_b = _compute_stats(model_b.polydata)

    if same_topology:
        topology_match, topo_diff = _compare_topology(
            model_a.polydata, model_b.polydata
        )
        distances = _compute_exact_vertex_distances(
            model_a.polydata, model_b.polydata
        )
    else:
        topology_match = False
        topo_diff = [
            "拓扑未做精确对比 (same_topology=False)",
            f"顶点数: A={stats_a.vertex_count}, B={stats_b.vertex_count}",
            f"三角面数: A={stats_a.triangle_count}, B={stats_b.triangle_count}",
        ]
        distances = _compute_nearest_neighbor_distances(
            model_a.polydata, model_b.polydata
        )

    p10, p25, p50, p75, p90 = np.percentile(distances, [10, 25, 50, 75, 90])
    p95, p99 = np.percentile(distances, [95, 99])

    overlap_ratio = float(np.sum(distances <= tolerance)) / len(distances)

    return STLComparison(
        stats_a=stats_a,
        stats_b=stats_b,
        topology_match=topology_match,
        topology_differences=topo_diff,
        vertex_distances=distances,
        max_distance=float(np.max(distances)),
        mean_distance=float(np.mean(distances)),
        rms_distance=float(np.sqrt(np.mean(distances**2))),
        distance_percentiles=(p10, p25, p50, p75, p90, p95, p99),
        overlap_ratio=overlap_ratio,
    )


def print_comparison_report(report: STLComparison,
                            label_a: str = "模型A",
                            label_b: str = "模型B"):
    """打印对比报告到控制台"""
    print("=" * 60)
    print("STL 文件结构对比报告")
    print("=" * 60)

    s = report.stats_a
    print(f"\n{label_a}:")
    print(f"  顶点数:   {s.vertex_count:,}")
    print(f"  三角面数: {s.triangle_count:,}")
    print(f"  体积:     {s.volume:,.2f} mm³")
    print(f"  表面积:   {s.surface_area:,.2f} mm²")
    print(f"  质心:     ({s.centroid[0]:.2f}, {s.centroid[1]:.2f}, {s.centroid[2]:.2f})")
    print(f"  尺寸:     {s.extent[0]:.2f} × {s.extent[1]:.2f} × {s.extent[2]:.2f} mm")

    s = report.stats_b
    print(f"\n{label_b}:")
    print(f"  顶点数:   {s.vertex_count:,}")
    print(f"  三角面数: {s.triangle_count:,}")
    print(f"  体积:     {s.volume:,.2f} mm³")
    print(f"  表面积:   {s.surface_area:,.2f} mm²")
    print(f"  质心:     ({s.centroid[0]:.2f}, {s.centroid[1]:.2f}, {s.centroid[2]:.2f})")
    print(f"  尺寸:     {s.extent[0]:.2f} × {s.extent[1]:.2f} × {s.extent[2]:.2f} mm")

    print(f"\n{'─' * 60}")
    print("差异分析:")
    print(f"  顶点差:   {report.stats_b.vertex_count - report.stats_a.vertex_count:+,}")
    print(f"  面差:     {report.stats_b.triangle_count - report.stats_a.triangle_count:+,}")
    print(f"  体积差:   {report.stats_b.volume - report.stats_a.volume:+,.2f} mm³")
    print(f"  表面积差: {report.stats_b.surface_area - report.stats_a.surface_area:+,.2f} mm²")

    print(f"\n{'─' * 60}")
    print("拓扑对比:")
    print(f"  拓扑一致: {'是' if report.topology_match else '否'}")
    for diff in report.topology_differences[:5]:
        print(f"  - {diff}")

    print(f"\n{'─' * 60}")
    print("几何距离 (mm):")
    p10, p25, p50, p75, p90, p95, p99 = report.distance_percentiles
    print(f"  平均:  {report.mean_distance:.4f}")
    print(f"  RMS:   {report.rms_distance:.4f}")
    print(f"  最大:  {report.max_distance:.4f}")
    print(f"  P10:   {p10:.4f}  P25: {p25:.4f}  P50: {p50:.4f}")
    print(f"  P75:   {p75:.4f}  P90: {p90:.4f}  P95: {p95:.4f}")
    print(f"  P99:   {p99:.4f}")
    print(f"  重叠率 (公差内): {report.overlap_ratio * 100:.1f}%")
    print("=" * 60)


# ---- 内部函数 ----


def _compute_stats(polydata: vtk.vtkPolyData) -> MeshStats:
    """计算网格统计信息"""
    from vtk.util.numpy_support import vtk_to_numpy

    bounds = polydata.GetBounds()
    extent = np.array([
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    ])

    vertices = vtk_to_numpy(polydata.GetPoints().GetData())
    centroid = vertices.mean(axis=0)

    # 体积计算（基于三角面的体积积分）
    volume = _compute_volume(polydata)

    # 表面积
    surface_area = _compute_surface_area(polydata)

    return MeshStats(
        vertex_count=polydata.GetNumberOfPoints(),
        triangle_count=polydata.GetNumberOfCells(),
        bounds=(bounds[0], bounds[2], bounds[4],
                bounds[1], bounds[3], bounds[5]),
        extent=extent,
        centroid=centroid,
        volume=volume,
        surface_area=surface_area,
    )


def _compute_volume(polydata: vtk.vtkPolyData) -> float:
    """
    计算封闭网格的体积

    使用三角面体积积分公式:
    V = Σ (1/6) * (v1 × v2) · v3
    其中每个三角面与原点形成一个四面体
    """
    from vtk.util.numpy_support import vtk_to_numpy

    cells = polydata.GetPolys()
    cells.InitTraversal()
    id_list = vtk.vtkIdList()

    vertices = vtk_to_numpy(polydata.GetPoints().GetData())
    volume = 0.0

    while cells.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            v0 = vertices[id_list.GetId(0)]
            v1 = vertices[id_list.GetId(1)]
            v2 = vertices[id_list.GetId(2)]
            # 四面体体积 = (1/6) * |a · (b × c)|
            cross = np.cross(v1, v2)
            volume += np.dot(v0, cross) / 6.0

    return abs(volume)


def _compute_surface_area(polydata: vtk.vtkPolyData) -> float:
    """计算网格表面积"""
    from vtk.util.numpy_support import vtk_to_numpy

    cells = polydata.GetPolys()
    cells.InitTraversal()
    id_list = vtk.vtkIdList()

    vertices = vtk_to_numpy(polydata.GetPoints().GetData())
    area = 0.0

    while cells.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            v0 = vertices[id_list.GetId(0)]
            v1 = vertices[id_list.GetId(1)]
            v2 = vertices[id_list.GetId(2)]
            edge1 = v1 - v0
            edge2 = v2 - v0
            area += 0.5 * np.linalg.norm(np.cross(edge1, edge2))

    return area


def _compare_topology(poly_a: vtk.vtkPolyData,
                      poly_b: vtk.vtkPolyData) -> tuple:
    """
    对比两个网格的拓扑结构

    返回: (是否一致, 差异列表)
    """
    differences = []

    if poly_a.GetNumberOfPoints() != poly_b.GetNumberOfPoints():
        differences.append(
            f"顶点数不同: {poly_a.GetNumberOfPoints()} vs {poly_b.GetNumberOfPoints()}"
        )
    if poly_a.GetNumberOfCells() != poly_b.GetNumberOfCells():
        differences.append(
            f"三角面数不同: {poly_a.GetNumberOfCells()} vs {poly_b.GetNumberOfCells()}"
        )

    # 检查顶点坐标是否相同
    from vtk.util.numpy_support import vtk_to_numpy
    verts_a = vtk_to_numpy(poly_a.GetPoints().GetData())
    verts_b = vtk_to_numpy(poly_b.GetPoints().GetData())

    if verts_a.shape == verts_b.shape:
        max_vertex_diff = np.max(np.abs(verts_a - verts_b))
        if max_vertex_diff > 1e-6:
            differences.append(f"顶点坐标最大差异: {max_vertex_diff:.6f} mm")
    else:
        differences.append("顶点数组形状不同")

    # 检查面连接关系是否相同
    if poly_a.GetNumberOfCells() == poly_b.GetNumberOfCells():
        cells_a = _extract_cell_array(poly_a)
        cells_b = _extract_cell_array(poly_b)
        if not np.array_equal(cells_a, cells_b):
            differences.append("三角面连接关系不同")

    return len(differences) == 0, differences


def _extract_cell_array(polydata: vtk.vtkPolyData) -> np.ndarray:
    """提取面连接关系为 (N_tri, 3) 数组"""
    cells = polydata.GetPolys()
    cells.InitTraversal()
    id_list = vtk.vtkIdList()

    cell_list = []
    while cells.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            cell_list.append([
                id_list.GetId(0),
                id_list.GetId(1),
                id_list.GetId(2),
            ])

    return np.array(cell_list, dtype=np.int64)


def _compute_exact_vertex_distances(poly_a: vtk.vtkPolyData,
                                    poly_b: vtk.vtkPolyData) -> np.ndarray:
    """
    精确顶点对比（相同拓扑）

    假设两个网格有相同的顶点数和相同的顶点索引顺序
    返回每个对应顶点的距离
    """
    from vtk.util.numpy_support import vtk_to_numpy
    verts_a = vtk_to_numpy(poly_a.GetPoints().GetData())
    verts_b = vtk_to_numpy(poly_b.GetPoints().GetData())
    return np.linalg.norm(verts_a - verts_b, axis=1)


def _compute_nearest_neighbor_distances(poly_a: vtk.vtkPolyData,
                                        poly_b: vtk.vtkPolyData) -> np.ndarray:
    """
    最近邻距离分析（不同拓扑）

    对 A 的每个顶点，在 B 上找最近的顶点，返回距离数组
    同时对 B 的每个顶点在 A 上找最近顶点
    取两者的并集作为整体距离
    """
    from vtk.util.numpy_support import vtk_to_numpy

    verts_a = vtk_to_numpy(poly_a.GetPoints().GetData())
    verts_b = vtk_to_numpy(poly_b.GetPoints().GetData())

    tree_b = cKDTree(verts_b)
    dist_a_to_b, _ = tree_b.query(verts_a, k=1)

    tree_a = cKDTree(verts_a)
    dist_b_to_a, _ = tree_a.query(verts_b, k=1)

    return np.concatenate([dist_a_to_b, dist_b_to_a])
