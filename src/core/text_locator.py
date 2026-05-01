"""STL 网格表面文字标记定位 — 基于局部曲率异常

原理：文字标记会在光滑表面上产生局部凹凸。
使用 VTK 的 vtkCurvatures 计算高斯曲率，找出曲率异常的连通区域。
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from dataclasses import dataclass


@dataclass
class TextRegion:
    center: np.ndarray
    extent: np.ndarray
    vertex_indices: np.ndarray
    score: float


def locate_text_on_model(filepath: str):
    """定位 STL 模型表面的文字标记区域"""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filepath)
    reader.Update()
    poly = reader.GetOutput()

    n_pts = poly.GetNumberOfPoints()
    bounds = poly.GetBounds()
    print(f"加载: {filepath}")
    print(f"顶点: {n_pts:,}, 面: {poly.GetNumberOfCells():,}")
    print(f"包围盒: X[{bounds[0]:.1f},{bounds[1]:.1f}] "
          f"Y[{bounds[2]:.1f},{bounds[3]:.1f}] "
          f"Z[{bounds[4]:.1f},{bounds[5]:.1f}]")

    vertices = vtk_to_numpy(poly.GetPoints().GetData())

    # ---- 方法 1: 计算每个三角面的法向量，找相邻面法向变化大的区域 ----
    # 文字区域会有更多法向突变
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(poly)
    normals_filter.ComputeCellNormalsOn()
    normals_filter.ComputePointNormalsOff()
    normals_feature = vtk.vtkFeatureEdges()
    normals_feature.SetInputData(poly)
    normals_feature.BoundaryEdgesOff()
    normals_feature.FeatureEdgesOn()
    normals_feature.SetFeatureAngle(30)
    normals_feature.NonManifoldEdgesOff()
    normals_feature.Update()

    feature_edges = normals_feature.GetOutput()
    n_feature_edges = feature_edges.GetNumberOfCells()
    print(f"\n特征边(30度角): {n_feature_edges}")

    if n_feature_edges > 0:
        feature_pts = vtk_to_numpy(feature_edges.GetPoints().GetData())
        print(f"特征边顶点: {len(feature_pts):,}")

        # 按空间位置聚类特征边顶点
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        # 用网格大小决定聚类阈值
        model_size = np.ptp(vertices, axis=0).max()
        cluster_threshold = model_size * 0.03  # 3% 模型尺寸

        # 快速聚类：用 cKDTree 做密度聚类
        from scipy.spatial import cKDTree
        tree = cKDTree(feature_pts)

        # DBSCAN 手动实现
        visited = np.zeros(len(feature_pts), dtype=bool)
        labels = -np.ones(len(feature_pts), dtype=int)
        cluster_id = 0

        # 找 k 近邻距离作为 eps
        dists, _ = tree.query(feature_pts, k=10)
        eps = np.median(dists) * 5
        eps = min(eps, cluster_threshold)

        for i in range(len(feature_pts)):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = tree.query_ball_point(feature_pts[i], r=eps)
            if len(neighbors) < 5:
                continue
            # 扩展集群
            cluster_points = [i]
            j = 0
            while j < len(cluster_points):
                idx = cluster_points[j]
                j += 1
                if visited[idx]:
                    continue
                visited[idx] = True
                more_neighbors = tree.query_ball_point(feature_pts[idx], r=eps)
                if len(more_neighbors) >= 5:
                    cluster_points.extend(more_neighbors)

            if len(cluster_points) >= 20:
                pts = feature_pts[cluster_points]
                center = pts.mean(axis=0)
                extent = np.ptp(pts, axis=0)
                rx = (center[0] - bounds[0]) / max(bounds[1] - bounds[0], 1e-10) * 100
                ry = (center[1] - bounds[2]) / max(bounds[3] - bounds[2], 1e-10) * 100
                rz = (center[2] - bounds[4]) / max(bounds[5] - bounds[4], 1e-10) * 100

                print(f"\n  特征集群 {cluster_id+1}: "
                      f"中心=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f}) "
                      f"尺寸=({extent[0]:.1f},{extent[1]:.1f},{extent[2]:.1f})mm "
                      f"点数={len(cluster_points)} "
                      f"位置=({rx:.0f}%,{ry:.0f}%,{rz:.0f}%)")
                cluster_id += 1

    # ---- 方法 2: 表面法向异常检测 ----
    # 在平坦表面上，文字区域的法向会显著偏离周围
    print("\n--- 表面法向异常检测 ---")

    # 计算点法线
    norm_calc = vtk.vtkPolyDataNormals()
    norm_calc.SetInputData(poly)
    norm_calc.ComputePointNormalsOn()
    norm_calc.ComputeCellNormalsOff()
    norm_calc.SplittingOff()
    norm_calc.ConsistencyOn()
    norm_calc.Update()

    pt_normals = vtk_to_numpy(norm_calc.GetOutput().GetPointData().GetNormals())

    # 对每个顶点，计算其与 k 近邻的平均法向偏差
    k = 20
    tree = cKDTree(vertices)
    _, indices = tree.query(vertices, k=k+1)
    neighbor_indices = indices[:, 1:]  # 排除自身

    # 法向点积平均值 -> 1=完全一致, <1=有曲率
    neighbor_normals = pt_normals[neighbor_indices]  # (N, k, 3)
    dot_products = np.sum(pt_normals[:, np.newaxis, :] * neighbor_normals, axis=2)  # (N, k)
    mean_dot = np.mean(dot_products, axis=1)  # (N,)

    # 法向异常 = 1 - mean_dot
    normal_variation = 1.0 - mean_dot

    # 高异常区域
    threshold = np.percentile(normal_variation, 99.5)
    high_var_mask = normal_variation > threshold
    high_var_indices = np.where(high_var_mask)[0]

    print(f"法向异常范围: [{normal_variation.min():.6f}, {normal_variation.max():.6f}]")
    print(f"P99.5 阈值: {threshold:.6f}")
    print(f"高异常顶点: {len(high_var_indices):,}")

    if len(high_var_indices) > 100:
        # 空间聚类
        high_var_pts = vertices[high_var_indices]
        tree2 = cKDTree(high_var_pts)

        # 模型局部距离
        dists2, _ = tree2.query(high_var_pts, k=5)
        eps2 = np.median(dists2) * 8

        visited2 = np.zeros(len(high_var_pts), dtype=bool)
        cluster_count = 0

        for i in range(len(high_var_pts)):
            if visited2[i]:
                continue
            visited2[i] = True
            neighbors2 = tree2.query_ball_point(high_var_pts[i], r=eps2)
            if len(neighbors2) < 3:
                continue

            cluster_pts = [i]
            j = 0
            while j < len(cluster_pts):
                idx = cluster_pts[j]
                j += 1
                if visited2[idx]:
                    continue
                visited2[idx] = True
                more = tree2.query_ball_point(high_var_pts[idx], r=eps2)
                if len(more) >= 3:
                    cluster_pts.extend(more)

            if len(cluster_pts) >= 50:
                pts = high_var_pts[cluster_pts]
                center = pts.mean(axis=0)
                extent = np.ptp(pts, axis=0)
                rx = (center[0] - bounds[0]) / max(bounds[1] - bounds[0], 1e-10) * 100
                ry = (center[1] - bounds[2]) / max(bounds[3] - bounds[2], 1e-10) * 100
                rz = (center[2] - bounds[4]) / max(bounds[5] - bounds[4], 1e-10) * 100

                print(f"  异常集群 {cluster_count+1}: "
                      f"中心=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f}) "
                      f"尺寸=({extent[0]:.1f},{extent[1]:.1f},{extent[2]:.1f})mm "
                      f"顶点={len(cluster_pts):,} "
                      f"位置=({rx:.0f}%,{ry:.0f}%,{rz:.0f}%)")
                cluster_count += 1

        if cluster_count == 0:
            print("  未找到足够大的异常聚类")

    # ---- 方法 3: 保存残差/法向可视化 STL ----
    print("\n--- 保存可视化 ---")

    # 归一化法向异常
    nv_min, nv_max = normal_variation.min(), normal_variation.max()
    if nv_max > nv_min:
        nv_norm = (normal_variation - nv_min) / (nv_max - nv_min)
    else:
        nv_norm = np.zeros_like(normal_variation)

    output = vtk.vtkPolyData()
    output.DeepCopy(poly)
    scalar_array = numpy_to_vtk(nv_norm)
    scalar_array.SetName("NormalVariation")
    output.GetPointData().SetScalars(scalar_array)

    out_path = filepath.rsplit(".", 1)[0] + "_text_map.stl"
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(output)
    writer.Write()
    print(f"法向异常可视化: {out_path}")
    print("\n用法：在 AnklePro 中加载原模型 + 此可视化文件，"
          "颜色亮的区域即为可能的文字标记位置。")


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/qb_LEFT_exol.stl"
    locate_text_on_model(filepath)
