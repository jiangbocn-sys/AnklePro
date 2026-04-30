"""VTK 场景管理 — 模型渲染、坐标轴、颜色映射、标量条"""

from typing import Optional, Tuple

import numpy as np
import vtk

from src.config import (
    BACKGROUND_COLOR,
    BRACE_COLOR,
    FOOT_COLOR,
    MODEL_OPACITY_NORMAL,
    MODEL_OPACITY_RESULT,
    AXES_LENGTH,
)
from src.render.scene_overlay import SceneOverlay


class SceneManager(SceneOverlay):
    """VTK 渲染场景管理 — 模型、变换、颜色映射、坐标轴"""

    def __init__(self, render_window: vtk.vtkRenderWindow):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*BACKGROUND_COLOR)
        render_window.AddRenderer(self.renderer)

        self.axes_actor: Optional[vtk.vtkAxesActor] = None
        self.axes_length = AXES_LENGTH

        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetTitle("Distance (mm)")
        self.scalar_bar.SetNumberOfLabels(6)
        self.scalar_bar.VisibilityOff()

        self._lut = vtk.vtkLookupTable()
        self._lut.SetNumberOfTableValues(256)
        self._lut.Build()
        self.scalar_bar.SetLookupTable(self._lut)
        self.renderer.AddActor2D(self.scalar_bar)

        self._foot_actor: Optional[vtk.vtkActor] = None
        self._brace_actor: Optional[vtk.vtkActor] = None

        self._current_brace_matrix: Optional[np.ndarray] = None
        self._inner_surface_actors: list = []

        super().__init__()

    # ---- 模型添加 ----

    def add_foot_model(self, polydata: vtk.vtkPolyData) -> vtk.vtkActor:
        """添加足部模型到场景"""
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*FOOT_COLOR)
        actor.GetProperty().SetOpacity(MODEL_OPACITY_NORMAL)
        actor.GetProperty().SetRepresentationToSurface()

        self.renderer.AddActor(actor)
        self._foot_actor = actor
        return actor

    def add_brace_model(self, polydata: vtk.vtkPolyData) -> vtk.vtkActor:
        """添加护具模型到场景"""
        self._current_brace_matrix = np.eye(4, dtype=np.float64)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*BRACE_COLOR)
        actor.GetProperty().SetOpacity(MODEL_OPACITY_NORMAL)
        actor.GetProperty().SetRepresentationToSurface()

        self.renderer.AddActor(actor)
        self._brace_actor = actor
        return actor

    # ---- 变换应用 ----

    def apply_transform(self, matrix: np.ndarray):
        """应用4x4变换矩阵到护具模型"""
        if self._brace_actor is None:
            return

        transform = vtk.vtkTransform()
        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, float(matrix[i, j]))
        transform.SetMatrix(vtk_matrix)

        self._brace_actor.SetUserTransform(transform)
        self._current_brace_matrix = matrix.copy()

        for actor in self._inner_surface_actors:
            vtk_matrix = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    vtk_matrix.SetElement(i, j, float(matrix[i, j]))
            actor.SetUserMatrix(vtk_matrix)

    # ---- 颜色映射 ----

    def apply_distance_colors(
        self,
        brace_polydata: vtk.vtkPolyData,
        distances: np.ndarray,
        thresholds: list,
    ):
        """将距离值映射为顶点颜色并更新护具 Actor"""
        if self._brace_actor is None:
            return

        colored_polydata = vtk.vtkPolyData()
        colored_polydata.DeepCopy(brace_polydata)

        n_points = colored_polydata.GetNumberOfPoints()
        color_array = vtk.vtkUnsignedCharArray()
        color_array.SetName("DistanceColors")
        color_array.SetNumberOfComponents(3)

        if len(distances) != n_points:
            for _ in range(n_points):
                color_array.InsertNextTuple3(
                    int(0.5 * 255), int(0.5 * 255), int(0.5 * 255)
                )
        else:
            for i in range(n_points):
                d = distances[i]
                rgb = (0.5, 0.5, 0.5)
                for t_min, t_max, color, _label in thresholds:
                    if t_min <= d < t_max:
                        rgb = color
                        break
                color_array.InsertNextTuple3(
                    int(rgb[0] * 255),
                    int(rgb[1] * 255),
                    int(rgb[2] * 255),
                )

        colored_polydata.GetPointData().SetScalars(color_array)

        d_min = float(np.min(distances))
        d_max = float(np.max(distances))
        self._lut.SetTableRange(d_min, d_max)

        n_colors = 256
        for i in range(n_colors):
            val = d_min + (d_max - d_min) * i / (n_colors - 1)
            rgb = (0.5, 0.5, 0.5)
            for t_min, t_max, color, _label in thresholds:
                if t_min <= val < t_max:
                    rgb = color
                    break
            self._lut.SetTableValue(i, rgb[0], rgb[1], rgb[2], 1.0)
        self._lut.Modified()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(colored_polydata)
        mapper.SetScalarModeToUsePointData()
        mapper.SetScalarRange(d_min, d_max)
        mapper.SetLookupTable(self._lut)

        self._brace_actor.SetMapper(mapper)
        self._brace_actor.GetProperty().SetOpacity(MODEL_OPACITY_RESULT)

        self._update_scalar_bar(thresholds)

    def apply_inner_surface_colors(
        self,
        brace_polydata: vtk.vtkPolyData,
        inner_vertex_indices: np.ndarray,
        distances: np.ndarray,
        thresholds: list,
    ):
        """只对内侧面的顶点应用距离颜色，其余顶点保持原色"""
        if self._brace_actor is None:
            return

        colored_polydata = vtk.vtkPolyData()
        colored_polydata.DeepCopy(brace_polydata)

        n_total = colored_polydata.GetNumberOfPoints()
        color_array = vtk.vtkUnsignedCharArray()
        color_array.SetName("DistanceColors")
        color_array.SetNumberOfComponents(3)
        color_array.SetNumberOfTuples(n_total)

        dist_map = {}
        for i, vid in enumerate(inner_vertex_indices):
            dist_map[int(vid)] = distances[i]

        brace_color = (int(BRACE_COLOR[0] * 255), int(BRACE_COLOR[1] * 255), int(BRACE_COLOR[2] * 255))
        for i in range(n_total):
            if i in dist_map:
                d = dist_map[i]
                rgb = (0.5, 0.5, 0.5)
                for t_min, t_max, color, _label in thresholds:
                    if t_min <= d < t_max:
                        rgb = color
                        break
                color_array.SetTuple3(i, int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            else:
                color_array.SetTuple3(i, *brace_color)

        colored_polydata.GetPointData().SetScalars(color_array)

        d_min = float(np.min(distances))
        d_max = float(np.max(distances))
        self._lut.SetTableRange(d_min, d_max)
        n_colors = 256
        for i in range(n_colors):
            val = d_min + (d_max - d_min) * i / (n_colors - 1)
            rgb = (0.5, 0.5, 0.5)
            for t_min, t_max, color, _label in thresholds:
                if t_min <= val < t_max:
                    rgb = color
                    break
            self._lut.SetTableValue(i, rgb[0], rgb[1], rgb[2], 1.0)
        self._lut.Modified()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(colored_polydata)
        mapper.SetScalarModeToUsePointData()
        mapper.SetScalarRange(d_min, d_max)
        mapper.SetLookupTable(self._lut)

        self._brace_actor.SetMapper(mapper)
        self._brace_actor.GetProperty().SetOpacity(MODEL_OPACITY_RESULT)

        self._update_scalar_bar(thresholds)

    def _create_inner_surface_actor(
        self,
        polydata: vtk.vtkPolyData,
        inner_cells: np.ndarray,
        color: tuple = (100, 180, 255),
    ) -> vtk.vtkActor:
        """创建内侧面高亮显示的 Actor"""
        n_cells = polydata.GetNumberOfCells()
        color_array = vtk.vtkUnsignedCharArray()
        color_array.SetName("InnerSurfaceColors")
        color_array.SetNumberOfComponents(3)
        color_array.SetNumberOfTuples(n_cells)

        r, g, b = color
        gray_r, gray_g, gray_b = 60, 60, 70

        inner_set = set(inner_cells.tolist())
        for i in range(n_cells):
            if i in inner_set:
                color_array.SetTuple3(i, int(r), int(g), int(b))
            else:
                color_array.SetTuple3(i, gray_r, gray_g, gray_b)

        colored = vtk.vtkPolyData()
        colored.DeepCopy(polydata)
        colored.GetCellData().SetScalars(color_array)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(colored)
        mapper.SetScalarModeToUseCellData()
        mapper.SetScalarRange(0, 255)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.9)
        actor.GetProperty().LightingOff()

        if self._current_brace_matrix is not None:
            vtk_matrix = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    vtk_matrix.SetElement(i, j, float(self._current_brace_matrix[i, j]))
            actor.SetUserMatrix(vtk_matrix)

        self._inner_surface_actors.append(actor)
        return actor

    def clear_inner_surface_actors(self):
        """清除所有内侧面区域 Actor"""
        for actor in self._inner_surface_actors:
            self.renderer.RemoveActor(actor)
        self._inner_surface_actors.clear()

    def clear_distance_colors(self):
        """清除距离颜色，恢复护具原始外观"""
        if self._brace_actor is None:
            return
        self._brace_actor.GetProperty().SetColor(*BRACE_COLOR)
        self._brace_actor.GetProperty().SetOpacity(MODEL_OPACITY_NORMAL)
        self.scalar_bar.VisibilityOff()

    def set_wireframe(self, actor: vtk.vtkActor, enabled: bool):
        """切换线框/实体模式"""
        if actor:
            if enabled:
                actor.GetProperty().SetRepresentationToWireframe()
            else:
                actor.GetProperty().SetRepresentationToSurface()

    def set_foot_opacity(self, opacity: float):
        """设置足部模型透明度"""
        if self._foot_actor is None:
            return
        self._foot_actor.GetProperty().SetOpacity(opacity)
        self._foot_actor.GetProperty().SetRepresentationToWireframe() if opacity < 1.0 else self._foot_actor.GetProperty().SetRepresentationToSurface()

    # ---- 相机 ----

    def reset_camera(self):
        """重置相机到默认视角"""
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.Elevation(30)
        camera.Azimuth(45)
        self.renderer.ResetCamera()

    def fit_camera_to_models(self):
        """自动调整相机以适应所有模型"""
        self.renderer.ResetCamera()

    # ---- 坐标轴 ----

    def add_axes(self, bounds: Tuple[float, float, float, float, float, float]):
        """添加带刻度的坐标轴"""
        if self.axes_actor is not None:
            self.renderer.RemoveActor(self.axes_actor)
        if hasattr(self, '_label_actors'):
            for actor in self._label_actors:
                self.renderer.RemoveActor(actor)
        if hasattr(self, '_tick_actors'):
            for actor in self._tick_actors:
                self.renderer.RemoveActor(actor)

        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]

        axes_length = max(x_range, y_range, z_range) * 1.2
        self.axes_length = axes_length

        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ]

        self._create_axis_line(center, axes_length, 'X')
        self._create_axis_line(center, axes_length, 'Y')
        self._create_axis_line(center, axes_length, 'Z')
        self._add_axis_labels(center, axes_length)

    def _create_axis_line(self, center: list, axes_length: float, axis: str):
        """创建单个坐标轴线"""
        half_length = axes_length / 2

        if axis == 'X':
            p1 = [center[0] - half_length, center[1], center[2]]
            p2 = [center[0] + half_length, center[1], center[2]]
            color = (1, 0, 0)
        elif axis == 'Y':
            p1 = [center[0], center[1] - half_length, center[2]]
            p2 = [center[0], center[1] + half_length, center[2]]
            color = (0, 1, 0)
        else:
            p1 = [center[0], center[1], center[2] - half_length]
            p2 = [center[0], center[1], center[2] + half_length]
            color = (0, 0, 1)

        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(p1)
        line_source.SetPoint2(p2)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(line_source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetLineWidth(2)

        self.renderer.AddActor(actor)

        if not hasattr(self, '_axis_line_actors'):
            self._axis_line_actors = []
        self._axis_line_actors.append(actor)

    def _add_axis_labels(self, center: list, axes_length: float):
        """添加自定义坐标轴标签"""
        if hasattr(self, '_label_actors'):
            for actor in self._label_actors:
                self.renderer.RemoveActor(actor)

        self._label_actors = []

        for axis_name, pos_offset, color in [
            ("X", [axes_length/2 + 3, 0, 0], (1, 0, 0)),
            ("Y", [0, axes_length/2 + 3, 0], (0, 1, 0)),
            ("Z", [0, 0, axes_length/2 + 3], (0, 0, 1)),
        ]:
            label = vtk.vtkVectorText()
            label.SetText(axis_name)
            label_mapper = vtk.vtkPolyDataMapper()
            label_mapper.SetInputConnection(label.GetOutputPort())
            label_actor = vtk.vtkFollower()
            label_actor.SetMapper(label_mapper)
            label_actor.SetPosition(
                center[0] + pos_offset[0],
                center[1] + pos_offset[1],
                center[2] + pos_offset[2],
            )
            label_actor.GetProperty().SetColor(*color)
            label_actor.SetScale(0.3, 0.3, 0.3)
            self._label_actors.append(label_actor)
            self.renderer.AddActor(label_actor)

    # ---- 内部方法 ----

    def _update_scalar_bar(self, thresholds: list):
        """更新标量条显示"""
        self.scalar_bar.VisibilityOn()
        self.scalar_bar.SetNumberOfLabels(min(len(thresholds), 10))
        self.scalar_bar.SetTitle("Distance (mm)")
