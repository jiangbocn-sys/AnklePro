"""模型对比与仿射变换映射

解决两台 3D 扫描仪对同一物体扫描结果不一致的问题。
通过 ICP 对齐 + 仿射变换（12 参数）建立两个模型间的映射关系。

用法：
    from src.core.model_mapper import compute_mapping
    result = compute_mapping('foot_a.stl', 'foot_b.stl')
    print(result.matrix_4x4)
    print(f"平均残差: {result.mean_residual:.4f} mm")
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.spatial import cKDTree

from src.core.model_loader import load_stl


@dataclass
class AlignmentResult:
    """ICP 刚体对齐结果"""
    rotation: np.ndarray       # 3×3 旋转矩阵
    translation: np.ndarray    # 3×1 平移向量
    matrix_4x4: np.ndarray     # 4×4 刚体变换矩阵
    mean_error: float          # 平均对齐误差 (mm)
    iterations: int            # ICP 迭代次数


@dataclass
class ModelComparison:
    """两个模型的对比结果"""
    stats_a: dict = field(default_factory=dict)
    stats_b: dict = field(default_factory=dict)
    alignment: Optional[AlignmentResult] = None
    affine_matrix: Optional[np.ndarray] = None  # 4×4 仿射矩阵
    residuals: Optional[np.ndarray] = None       # 逐顶点残差
    mean_residual: float = 0.0
    max_residual: float = 0.0
    rms_residual: float = 0.0
    percentile_residuals: tuple = (0.0,) * 7


def _compute_stats(polydata: vtk.vtkPolyData) -> dict:
    """计算网格统计信息"""
    verts = vtk_to_numpy(polydata.GetPoints().GetData())
    bounds = polydata.GetBounds()
    extent = np.array([
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    ])
    return {
        "vertex_count": polydata.GetNumberOfPoints(),
        "triangle_count": polydata.GetNumberOfCells(),
        "bounds": bounds,
        "extent": extent,
        "centroid": verts.mean(axis=0),
    }


def _icp_alignment(source: np.ndarray, target: np.ndarray,
                   max_iter: int = 50, tol: float = 1e-6) -> AlignmentResult:
    """
    ICP (Iterative Closest Point) 刚体对齐

    用 SVD 方法求解最优旋转 + 平移，迭代更新。
    """
    tree = cKDTree(target)
    n = len(source)

    # 初始化：恒等变换
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)

    current = source.astype(np.float64).copy()

    for iteration in range(max_iter):
        # 找最近邻
        dists, indices = tree.query(current, k=1)
        matched = target[indices].astype(np.float64)

        mean_error = float(np.mean(dists))
        if iteration > 0 and abs(mean_error - prev_error) < tol:
            break
        prev_error = mean_error

        # 去中心化
        src_mean = current.mean(axis=0)
        tgt_mean = matched.mean(axis=0)
        A = current - src_mean
        B = matched - tgt_mean

        # SVD 求解最优旋转
        H = np.dot(A.T, B)
        U, S, Vt = np.linalg.svd(H)
        R_inc = np.dot(Vt.T, U.T)

        # 保证右手系（det = 1）
        if np.linalg.det(R_inc) < 0:
            Vt[-1, :] *= -1
            R_inc = np.dot(Vt.T, U.T)

        # 平移
        t_inc = tgt_mean - np.dot(R_inc, src_mean)

        # 累积变换
        R_total = np.dot(R_inc, R_total)
        t_total = np.dot(R_inc, t_total) + t_inc

        # 更新当前点
        current = np.dot(R_inc, current.T).T + t_inc

    return AlignmentResult(
        rotation=R_total, translation=t_total,
        matrix_4x4=_make_4x4(R_total, t_total),
        mean_error=prev_error, iterations=iteration + 1,
    )


def _make_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """构建 4×4 变换矩阵"""
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def _compute_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    计算仿射变换矩阵（12 参数）

    用最近邻匹配找到对应点，然后用最小二乘法求解仿射矩阵。
    求解 M 使得 source @ M.T ≈ matched_target
    """
    tree = cKDTree(target)
    dists, indices = tree.query(source, k=1)
    matched = target[indices]

    n = len(source)
    A = np.hstack([source, np.ones((n, 1))])  # (N, 4)

    X, _, _, _ = np.linalg.lstsq(A, matched, rcond=None)

    M = np.eye(4)
    M[:3, :] = X.T
    return M


def compute_mapping(filepath_a: str, filepath_b: str,
                    max_icp_iter: int = 50) -> ModelComparison:
    """
    计算两个模型间的仿射变换映射

    步骤：
    1. 加载两个模型
    2. ICP 刚体对齐（消除位置和旋转差异）
    3. 在对齐基础上计算仿射变换（含缩放 + 剪切）
    4. 计算残差分布
    """
    model_a = load_stl(filepath_a)
    model_b = load_stl(filepath_b)

    verts_a = vtk_to_numpy(model_a.polydata.GetPoints().GetData()).astype(np.float64)
    verts_b = vtk_to_numpy(model_b.polydata.GetPoints().GetData()).astype(np.float64)

    comparison = ModelComparison()
    comparison.stats_a = _compute_stats(model_a.polydata)
    comparison.stats_b = _compute_stats(model_b.polydata)

    # Step 1: ICP 刚体对齐
    alignment = _icp_alignment(verts_a, verts_b, max_iter=max_icp_iter)
    comparison.alignment = alignment

    # Step 2: 对 A 应用刚体变换
    verts_a_aligned = (alignment.rotation @ verts_a.T).T + alignment.translation

    # Step 3: 计算仿射变换（缩放 + 剪切）
    affine = _compute_affine(verts_a_aligned, verts_b)
    comparison.affine_matrix = affine

    # Step 4: 计算残差（对原始 A 应用完整变换：ICP 刚体 + 仿射缩放）
    M_combined = alignment.matrix_4x4 @ affine
    verts_transformed = (M_combined[:3, :3] @ verts_a.T).T + M_combined[:3, 3]
    tree = cKDTree(verts_b)
    dists, _ = tree.query(verts_transformed, k=1)

    comparison.residuals = dists
    comparison.mean_residual = float(np.mean(dists))
    comparison.max_residual = float(np.max(dists))
    comparison.rms_residual = float(np.sqrt(np.mean(dists**2)))
    comparison.percentile_residuals = tuple(
        float(np.percentile(dists, p))
        for p in [10, 25, 50, 75, 90, 95, 99]
    )

    return comparison


def _apply_affine(verts: np.ndarray, matrix_3x4: np.ndarray) -> np.ndarray:
    """应用仿射变换到顶点集"""
    n = len(verts)
    homogeneous = np.hstack([verts, np.ones((n, 1))])
    return (homogeneous @ matrix_3x4)


def apply_transform_to_polydata(
    polydata: vtk.vtkPolyData,
    matrix_4x4: np.ndarray,
) -> vtk.vtkPolyData:
    """
    对 vtkPolyData 应用 4×4 仿射变换，返回新的 polydata
    """
    transform = vtk.vtkTransform()
    for i in range(4):
        for j in range(4):
            transform.GetMatrix().SetElement(i, j, float(matrix_4x4[i, j]))

    filter_ = vtk.vtkTransformPolyDataFilter()
    filter_.SetTransform(transform)
    filter_.SetInputData(polydata)
    filter_.Update()

    return filter_.GetOutput()


def export_transformed_stl(
    source_filepath: str,
    output_filepath: str,
    matrix_4x4: np.ndarray,
):
    """加载源 STL，应用变换，导出为 Binary STL"""
    model = load_stl(source_filepath)
    new_poly = apply_transform_to_polydata(model.polydata, matrix_4x4)

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_filepath)
    writer.SetFileTypeToBinary()
    writer.SetInputData(new_poly)
    writer.Write()


def print_mapping_report(comparison: ModelComparison,
                         label_a: str = "模型 A",
                         label_b: str = "模型 B"):
    """打印映射分析报告"""
    print("=" * 60)
    print("模型对比与映射分析报告")
    print("=" * 60)

    s = comparison.stats_a
    print(f"\n{label_a}:")
    print(f"  顶点数:   {s['vertex_count']:,}")
    print(f"  三角面数: {s['triangle_count']:,}")
    print(f"  质心:     ({s['centroid'][0]:.1f}, {s['centroid'][1]:.1f}, {s['centroid'][2]:.1f})")
    print(f"  尺寸:     {s['extent'][0]:.1f} × {s['extent'][1]:.1f} × {s['extent'][2]:.1f} mm")

    s = comparison.stats_b
    print(f"\n{label_b}:")
    print(f"  顶点数:   {s['vertex_count']:,}")
    print(f"  三角面数: {s['triangle_count']:,}")
    print(f"  质心:     ({s['centroid'][0]:.1f}, {s['centroid'][1]:.1f}, {s['centroid'][2]:.1f})")
    print(f"  尺寸:     {s['extent'][0]:.1f} × {s['extent'][1]:.1f} × {s['extent'][2]:.1f} mm")

    # ICP 对齐
    a = comparison.alignment
    print(f"\nICP 刚体对齐:")
    print(f"  迭代次数: {a.iterations}")
    print(f"  对齐误差: {a.mean_error:.4f} mm")
    print(f"  旋转矩阵:")
    print(f"    [{a.rotation[0,0]:.4f}, {a.rotation[0,1]:.4f}, {a.rotation[0,2]:.4f}]")
    print(f"    [{a.rotation[1,0]:.4f}, {a.rotation[1,1]:.4f}, {a.rotation[1,2]:.4f}]")
    print(f"    [{a.rotation[2,0]:.4f}, {a.rotation[2,1]:.4f}, {a.rotation[2,2]:.4f}]")
    print(f"  平移:     ({a.translation[0]:.2f}, {a.translation[1]:.2f}, {a.translation[2]:.2f}) mm")

    # 仿射矩阵
    M = comparison.affine_matrix
    print(f"\n仿射变换矩阵 (12 参数):")
    for i in range(4):
        print(f"  [{M[i,0]:.6f}, {M[i,1]:.6f}, {M[i,2]:.6f}, {M[i,3]:.6f}]")

    # 分解仿射矩阵（旋转 + 缩放 + 剪切）
    R_affine = M[:3, :3]
    t_affine = M[:3, 3]
    scales = np.linalg.norm(R_affine, axis=1)
    print(f"\n仿射分解:")
    print(f"  各轴缩放: X={scales[0]:.4f}  Y={scales[1]:.4f}  Z={scales[2]:.4f}")
    print(f"  平移:     ({t_affine[0]:.2f}, {t_affine[1]:.2f}, {t_affine[2]:.2f}) mm")

    # 残差
    p = comparison.percentile_residuals
    print(f"\n变换残差 (mm):")
    print(f"  平均:  {comparison.mean_residual:.4f}")
    print(f"  RMS:   {comparison.rms_residual:.4f}")
    print(f"  最大:  {comparison.max_residual:.4f}")
    print(f"  P10:   {p[0]:.4f}  P25: {p[1]:.4f}  P50: {p[2]:.4f}")
    print(f"  P75:   {p[3]:.4f}  P90: {p[4]:.4f}  P95: {p[5]:.4f}")
    print(f"  P99:   {p[6]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    if len(sys.argv) >= 3:
        a, b = sys.argv[1], sys.argv[2]
    else:
        a = "data/jiangbo_LEFT_foot.stl"
        b = "data/qb_LEFT_foot.stl"

    print(f"计算映射: {a} → {b}")
    result = compute_mapping(a, b)
    print_mapping_report(result, a.split("/")[-1], b.split("/")[-1])
