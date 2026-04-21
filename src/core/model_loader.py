"""模型加载器 — STL -> vtkPolyData"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import vtk
from scipy.spatial import cKDTree


@dataclass
class ModelData:
    """加载后的模型数据"""
    polydata: vtk.vtkPolyData          # VTK几何数据
    filepath: str                      # 原始文件路径
    name: str                          # 模型名称
    centroid: np.ndarray               # 质心坐标 [x, y, z]
    vertex_count: int                  # 顶点数
    triangle_count: int                # 三角面数
    bounding_box: tuple                # (min_x, min_y, min_z, max_x, max_y, max_z)
    kd_tree: Optional[cKDTree] = None  # 足部模型的空间索引


def load_stl(filepath: str) -> ModelData:
    """
    加载STL文件，返回ModelData

    自动检测ASCII/Binary格式。
    加载后执行：
    1. vtkCleanPolyData — 合并重复顶点
    2. vtkPolyDataNormals — 重新计算法线
    3. 计算包围盒和质心
    """
    filepath = str(Path(filepath).resolve())
    name = Path(filepath).stem

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filepath)
    reader.Update()

    polydata = reader.GetOutput()

    # 清理重复顶点
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(polydata)
    clean.PointMergingOn()
    clean.Update()
    polydata = clean.GetOutput()

    # 重新计算法线
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    polydata = normals.GetOutput()

    # 计算统计信息
    bounds = polydata.GetBounds()
    bounding_box = (bounds[0], bounds[2], bounds[4],
                    bounds[1], bounds[3], bounds[5])

    vertices = _polydata_to_vertices(polydata)
    centroid = vertices.mean(axis=0)

    vertex_count = polydata.GetNumberOfPoints()
    triangle_count = polydata.GetNumberOfCells()

    return ModelData(
        polydata=polydata,
        filepath=filepath,
        name=name,
        centroid=centroid,
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        bounding_box=bounding_box,
    )


def build_kd_tree(model: ModelData) -> ModelData:
    """为足部模型构建cKDTree空间索引"""
    vertices = _polydata_to_vertices(model.polydata)
    model.kd_tree = cKDTree(vertices)
    return model


def _polydata_to_vertices(polydata: vtk.vtkPolyData) -> np.ndarray:
    """提取vtkPolyData的顶点数组 (N, 3)"""
    points = polydata.GetPoints()
    n = points.GetNumberOfPoints()
    vertices = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        vertices[i] = points.GetPoint(i)
    return vertices
