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


class SceneManager:
    """VTK 渲染场景管理"""

    def __init__(self, render_window: vtk.vtkRenderWindow):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*BACKGROUND_COLOR)
        render_window.AddRenderer(self.renderer)

        # 坐标轴 - 初始化为默认长度，后续会根据模型动态调整
        self.axes_actor: Optional[vtk.vtkAxesActor] = None
        self.axes_length = AXES_LENGTH

        # 标量条（颜色图例）— 初始隐藏
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetTitle("Distance (mm)")
        self.scalar_bar.SetNumberOfLabels(6)
        self.scalar_bar.VisibilityOff()

        # 为标量条创建 LookupTable（避免警告）
        self._lut = vtk.vtkLookupTable()
        self._lut.SetNumberOfTableValues(256)
        self._lut.Build()
        self.scalar_bar.SetLookupTable(self._lut)

        self.renderer.AddActor2D(self.scalar_bar)

        # 模型 Actor
        self._foot_actor: Optional[vtk.vtkActor] = None
        self._brace_actor: Optional[vtk.vtkActor] = None

        # 操作说明文字覆盖层
        self._help_text: Optional[vtk.vtkTextActor] = None
        self._coord_text: Optional[vtk.vtkTextActor] = None

        # 护具变换管线（使用 vtkTransformFilter 实现变换）
        self._brace_transform_filter: Optional[vtk.vtkTransformPolyDataFilter] = None
        self._current_brace_matrix: Optional[np.ndarray] = None  # 当前变换矩阵
        self._inner_surface_actors: list = []  # 内侧面区域 Actor 列表
        self._measure_actors: list = []  # 测量可视化 Actor 组列表
        self._measure_status_text: Optional[vtk.vtkTextActor] = None
        self._cursor_text: Optional[vtk.vtkTextActor] = None
        self._pick_marker_actor: Optional[vtk.vtkActor] = None
        self._minmax_actors: list = []  # 最短/最长距离标示 Actor 列表

    def add_help_text(self, show: bool = True):
        """添加/隐藏操作说明文字（右下角）"""
        if show and self._help_text is None:
            self._help_text = vtk.vtkTextActor()
            help_content = (
                "视角控制:\n"
                "  ↑↓      缩放视角\n"
                "  ←→      旋转视角\n\n"
                "护具控制:\n"
                "  F/G     绕 Z 旋转 ±\n"
                "  H/J     绕 Y 旋转 ∓\n"
                "  N/M     绕 X 旋转 ∓\n"
                "  B/V     平移 X ±\n"
                "  U/I     平移 Y ±\n"
                "  R/T     平移 Z ±\n"
                "  D       线框/实体切换\n"
                "  ⌘+Z     撤销"
            )
            self._help_text.SetInput(help_content)
            self._help_text.GetTextProperty().SetFontSize(11)
            self._help_text.GetTextProperty().SetColor(1.0, 1.0, 1.0)
            self._help_text.GetTextProperty().SetOpacity(0.7)
            self._help_text.GetTextProperty().SetFontFamilyToCourier()
            self._help_text.SetPosition(10, 10)
            self.renderer.AddActor2D(self._help_text)
        elif self._help_text is not None:
            self._help_text.SetVisibility(1 if show else 0)

    def add_coord_text(self, text: str):
        """更新坐标信息文字（右上角）"""
        if self._coord_text is None:
            self._coord_text = vtk.vtkTextActor()
            self._coord_text.GetTextProperty().SetFontSize(13)
            self._coord_text.GetTextProperty().SetColor(1.0, 1.0, 0.3)
            self._coord_text.GetTextProperty().SetFontFamilyToCourier()
            self._coord_text.SetPosition(10, 500)
            self.renderer.AddActor2D(self._coord_text)
        self._coord_text.SetInput(text)

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
        # 使用 TransformFilter 管线实现变换
        self._brace_transform_filter = vtk.vtkTransformPolyDataFilter()
        self._brace_transform_filter.SetInputData(polydata)

        # 设置初始恒等变换
        identity = vtk.vtkTransform()
        identity.Identity()
        self._brace_transform_filter.SetTransform(identity)
        self._current_brace_matrix = np.eye(4, dtype=np.float64)

        self._brace_transform_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self._brace_transform_filter.GetOutputPort())

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
        """
        应用4x4变换矩阵到护具模型

        使用 vtkTransformPolyDataFilter 在 C++ 层执行变换，
        避免在 Python 中逐点循环。
        """
        if self._brace_transform_filter is None:
            return

        # 设置变换矩阵
        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, float(matrix[i, j]))

        transform = vtk.vtkTransform()
        transform.SetMatrix(vtk_matrix)
        self._brace_transform_filter.SetTransform(transform)
        # 关键：标记 filter 已修改，触发 VTK 管道更新
        self._brace_transform_filter.Modified()
        self._brace_transform_filter.Update()

        # 保存当前矩阵，用于内侧面高亮 Actor 的位置同步
        self._current_brace_matrix = matrix.copy()

        # 同步所有内侧面区域 Actor 的变换矩阵
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
        """
        将距离值映射为顶点颜色并更新护具 Actor

        注意: brace_polydata 是原始（未变换的）数据
        """
        if self._brace_actor is None:
            return

        # 深拷贝 polydata 以免修改原始几何
        colored_polydata = vtk.vtkPolyData()
        colored_polydata.DeepCopy(brace_polydata)

        n_points = colored_polydata.GetNumberOfPoints()
        color_array = vtk.vtkUnsignedCharArray()
        color_array.SetName("DistanceColors")
        color_array.SetNumberOfComponents(3)

        # 确保距离数组长度与顶点数一致
        if len(distances) != n_points:
            for _ in range(n_points):
                color_array.InsertNextTuple3(
                    int(0.5 * 255), int(0.5 * 255), int(0.5 * 255)
                )
        else:
            for i in range(n_points):
                d = distances[i]
                rgb = (0.5, 0.5, 0.5)  # 默认灰色
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

        # 如果当前有变换，需要将颜色数据也变换到当前位置
        if self._current_brace_matrix is not None:
            vtk_matrix = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    vtk_matrix.SetElement(i, j, float(self._current_brace_matrix[i, j]))
            transform = vtk.vtkTransform()
            transform.SetMatrix(vtk_matrix)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputData(colored_polydata)
            tf.SetTransform(transform)
            tf.Update()
            colored_polydata = tf.GetOutput()

        # 更新 lookup table 以支持标量条
        d_min = float(np.min(distances))
        d_max = float(np.max(distances))
        self._lut.SetTableRange(d_min, d_max)

        # 构建 lookup table 颜色
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

        # 更新 mapper（通过 transform filter）
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(colored_polydata)
        mapper.SetScalarModeToUsePointData()
        mapper.SetScalarRange(d_min, d_max)
        mapper.SetLookupTable(self._lut)

        self._brace_actor.SetMapper(mapper)
        self._brace_actor.GetProperty().SetOpacity(MODEL_OPACITY_RESULT)

        # 更新标量条
        self._update_scalar_bar(thresholds)

    def apply_inner_surface_colors(
        self,
        brace_polydata: vtk.vtkPolyData,
        inner_vertex_indices: np.ndarray,
        distances: np.ndarray,
        thresholds: list,
    ):
        """
        只对内侧面的顶点应用距离颜色，其余顶点保持原色

        参数:
            brace_polydata: 护具原始 vtkPolyData
            inner_vertex_indices: 内侧面顶点索引 (N,)
            distances: 内侧面顶点的逐顶点距离 (N,)
            thresholds: [(min, max, (R,G,B), label), ...]
        """
        if self._brace_actor is None:
            return

        # 深拷贝 polydata
        colored_polydata = vtk.vtkPolyData()
        colored_polydata.DeepCopy(brace_polydata)

        # 应用当前变换到几何体（确保颜色跟着走）
        if self._current_brace_matrix is not None:
            vtk_matrix = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    vtk_matrix.SetElement(i, j, float(self._current_brace_matrix[i, j]))
            transform = vtk.vtkTransform()
            transform.SetMatrix(vtk_matrix)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputData(colored_polydata)
            tf.SetTransform(transform)
            tf.Update()
            colored_polydata = tf.GetOutput()

        n_total = colored_polydata.GetNumberOfPoints()
        color_array = vtk.vtkUnsignedCharArray()
        color_array.SetName("DistanceColors")
        color_array.SetNumberOfComponents(3)
        color_array.SetNumberOfTuples(n_total)

        # 构建索引→距离映射
        dist_map = {}
        for i, vid in enumerate(inner_vertex_indices):
            dist_map[int(vid)] = distances[i]

        # 为每个顶点着色：内侧面顶点用距离颜色，其余用原色
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

        # 更新 lookup table
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

        # 更新 mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(colored_polydata)
        mapper.SetScalarModeToUsePointData()
        mapper.SetScalarRange(d_min, d_max)
        mapper.SetLookupTable(self._lut)

        self._brace_actor.SetMapper(mapper)
        self._brace_actor.GetProperty().SetOpacity(MODEL_OPACITY_RESULT)

        # 更新标量条
        self._update_scalar_bar(thresholds)

    def _create_inner_surface_actor(
        self,
        polydata: vtk.vtkPolyData,
        inner_cells: np.ndarray,
        color: tuple = (100, 180, 255),
    ) -> vtk.vtkActor:
        """
        创建内侧面高亮显示的 Actor
        内侧面显示为指定高亮颜色，其余面大幅降低饱和度和亮度

        参数:
            polydata: 护具模型 vtkPolyData
            inner_cells: 内侧面 cell 索引数组
            color: 内侧面高亮颜色 (R, G, B), 默认淡蓝色
        """
        n_cells = polydata.GetNumberOfCells()
        color_array = vtk.vtkUnsignedCharArray()
        color_array.SetName("InnerSurfaceColors")
        color_array.SetNumberOfComponents(3)
        color_array.SetNumberOfTuples(n_cells)

        r, g, b = color
        # 非内侧面使用深灰色，降低对比度以突出内侧面
        gray_r, gray_g, gray_b = 60, 60, 70

        inner_set = set(inner_cells.tolist())
        for i in range(n_cells):
            if i in inner_set:
                # 内侧面使用高亮鲜艳颜色
                color_array.SetTuple3(i, int(r), int(g), int(b))
            else:
                # 非内侧面使用暗灰色，形成强烈对比
                color_array.SetTuple3(i, gray_r, gray_g, gray_b)

        # 深拷贝并着色
        colored = vtk.vtkPolyData()
        colored.DeepCopy(polydata)
        colored.GetCellData().SetScalars(color_array)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(colored)
        mapper.SetScalarModeToUseCellData()
        mapper.SetScalarRange(0, 255)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # 内侧面区域稍微降低透明度，增强立体感
        actor.GetProperty().SetOpacity(0.9)
        # 关闭光照影响，保持颜色一致性
        actor.GetProperty().LightingOff()

        # 应用当前变换矩阵（与护具模型保持一致）
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

    # ---- 测量工具 ----

    def add_measurement(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        distance: float,
    ):
        """
        添加测量可视化 Actor

        - 黄色球体在 P1 和 P2
        - 青色连线连接两点
        - 3D 坐标标签（vtkFollower，始终面向相机）
        - 中点处显示距离
        """
        actors = []

        # 球体标记
        for pt in [p1, p2]:
            src = vtk.vtkSphereSource()
            src.SetCenter(pt[0], pt[1], pt[2])
            src.SetRadius(1.5)
            src.SetThetaResolution(16)
            src.SetPhiResolution(16)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(src.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 1.0, 0.0)
            self.renderer.AddActor(actor)
            actors.append(actor)

        # 连线
        line_src = vtk.vtkLineSource()
        line_src.SetPoint1(p1[0], p1[1], p1[2])
        line_src.SetPoint2(p2[0], p2[1], p2[2])
        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputConnection(line_src.GetOutputPort())
        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(0.0, 1.0, 1.0)
        line_actor.GetProperty().SetLineWidth(2)
        self.renderer.AddActor(line_actor)
        actors.append(line_actor)

        # 3D 坐标标签
        for pt, label in [
            (p1, f"P1({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})"),
            (p2, f"P2({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})"),
        ]:
            text = vtk.vtkVectorText()
            text.SetText(label)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(text.GetOutputPort())
            follower = vtk.vtkFollower()
            follower.SetMapper(mapper)
            follower.SetPosition(pt[0], pt[1] + 5, pt[2])
            follower.GetProperty().SetColor(1.0, 1.0, 0.3)
            follower.SetScale(0.15, 0.15, 0.15)
            self.renderer.AddActor(follower)
            actors.append(follower)

        # 中点距离标签
        mid = (p1 + p2) / 2.0
        dist_text = vtk.vtkVectorText()
        dist_text.SetText(f"d = {distance:.2f} mm")
        dist_mapper = vtk.vtkPolyDataMapper()
        dist_mapper.SetInputConnection(dist_text.GetOutputPort())
        dist_follower = vtk.vtkFollower()
        dist_follower.SetMapper(dist_mapper)
        dist_follower.SetPosition(mid[0], mid[1] - 5, mid[2])
        dist_follower.GetProperty().SetColor(0.0, 1.0, 0.5)
        dist_follower.SetScale(0.25, 0.25, 0.25)
        self.renderer.AddActor(dist_follower)
        actors.append(dist_follower)

        self._measure_actors.append(actors)

    def clear_measurements(self):
        """清除所有测量可视化 Actor"""
        for actors in self._measure_actors:
            for actor in actors:
                self.renderer.RemoveActor(actor)
        self._measure_actors.clear()

    def add_measurement_status_text(self, text: str):
        """显示测量状态文字（屏幕中上方）"""
        if self._measure_status_text is None:
            self._measure_status_text = vtk.vtkTextActor()
            self._measure_status_text.GetTextProperty().SetFontSize(14)
            self._measure_status_text.GetTextProperty().SetColor(0.0, 1.0, 1.0)
            self._measure_status_text.GetTextProperty().SetBold(True)
            self._measure_status_text.GetTextProperty().SetFontFamilyToCourier()
            self._measure_status_text.SetPosition(300, 600)
            self.renderer.AddActor2D(self._measure_status_text)
        self._measure_status_text.SetInput(text)
        self._measure_status_text.SetVisibility(1)

    def hide_measurement_status_text(self):
        """隐藏测量状态文字"""
        if self._measure_status_text is not None:
            self._measure_status_text.SetVisibility(0)

    def show_cursor_text(self, world_pos: np.ndarray):
        """在屏幕右下角显示光标处 3D 坐标"""
        if self._cursor_text is None:
            self._cursor_text = vtk.vtkTextActor()
            self._cursor_text.GetTextProperty().SetFontSize(13)
            self._cursor_text.GetTextProperty().SetColor(1.0, 1.0, 0.5)
            self._cursor_text.GetTextProperty().SetBold(True)
            self._cursor_text.GetTextProperty().SetFontFamilyToCourier()
            self._cursor_text.SetPosition(10, 30)
            self.renderer.AddActor2D(self._cursor_text)
        self._cursor_text.SetInput(
            f"({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f}) mm"
        )
        self._cursor_text.SetVisibility(1)

    def hide_cursor_text(self):
        """隐藏光标坐标文字"""
        if self._cursor_text is not None:
            self._cursor_text.SetVisibility(0)

    def add_pick_marker(self, world_pos: np.ndarray):
        """在拾取点处添加标记球体"""
        self._remove_pick_marker()
        src = vtk.vtkSphereSource()
        src.SetCenter(world_pos[0], world_pos[1], world_pos[2])
        src.SetRadius(2.0)
        src.SetThetaResolution(16)
        src.SetPhiResolution(16)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.5, 0.0)
        self.renderer.AddActor(actor)
        self._pick_marker_actor = actor

    def clear_pick_marker(self):
        """清除拾取标记"""
        self._remove_pick_marker()

    def _remove_pick_marker(self):
        """内部：移除标记球体"""
        if self._pick_marker_actor is not None:
            self.renderer.RemoveActor(self._pick_marker_actor)
            self._pick_marker_actor = None

    # ---- 最短/最长距离标示 ----

    def add_min_max_indicators(
        self,
        min_pos: np.ndarray,
        min_val: float,
        max_pos: np.ndarray,
        max_val: float,
        min_foot_pos: np.ndarray,
        max_foot_pos: np.ndarray,
    ):
        """
        在场景中标示最短距离和最长距离点

        - 标示线从护具点连接到足部表面对应的最近点
        - 最短距离：品红色球体 + 连线（因为最短距离在屏幕上不易观测）
        - 最长距离：亮绿色球体 + 连线
        - 3D 文字标签显示距离值
        """
        self.clear_min_max_indicators()

        for brace_pt, foot_pt, val, color, label_prefix in [
            (min_pos, min_foot_pos, min_val, (1.0, 0.0, 1.0), "MIN"),  # 品红色
            (max_pos, max_foot_pos, max_val, (0.0, 1.0, 0.0), "MAX"),  # 亮绿色
        ]:
            # 1. 护具点球体标记
            src = vtk.vtkSphereSource()
            src.SetCenter(brace_pt[0], brace_pt[1], brace_pt[2])
            src.SetRadius(2.5)
            src.SetThetaResolution(16)
            src.SetPhiResolution(16)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(src.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            self.renderer.AddActor(actor)
            self._minmax_actors.append(actor)

            # 2. 连线：从护具点连接到足部最近点
            line_src = vtk.vtkLineSource()
            line_src.SetPoint1(brace_pt[0], brace_pt[1], brace_pt[2])
            line_src.SetPoint2(foot_pt[0], foot_pt[1], foot_pt[2])
            line_mapper = vtk.vtkPolyDataMapper()
            line_mapper.SetInputConnection(line_src.GetOutputPort())
            line_actor = vtk.vtkActor()
            line_actor.SetMapper(line_mapper)
            line_actor.GetProperty().SetColor(*color)
            line_actor.GetProperty().SetLineWidth(3)
            self.renderer.AddActor(line_actor)
            self._minmax_actors.append(line_actor)

            # 3. 足部最近点球体标记
            foot_src = vtk.vtkSphereSource()
            foot_src.SetCenter(foot_pt[0], foot_pt[1], foot_pt[2])
            foot_src.SetRadius(2.0)
            foot_src.SetThetaResolution(16)
            foot_src.SetPhiResolution(16)
            foot_mapper = vtk.vtkPolyDataMapper()
            foot_mapper.SetInputConnection(foot_src.GetOutputPort())
            foot_actor = vtk.vtkActor()
            foot_actor.SetMapper(foot_mapper)
            foot_actor.GetProperty().SetColor(*color)
            self.renderer.AddActor(foot_actor)
            self._minmax_actors.append(foot_actor)

            # 4. 3D 文字标签（连线中点偏移处）
            mid = (brace_pt + foot_pt) / 2.0
            text = vtk.vtkVectorText()
            text.SetText(f"{label_prefix}: {val:+.2f} mm")
            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text.GetOutputPort())
            follower = vtk.vtkFollower()
            follower.SetMapper(text_mapper)
            follower.SetPosition(mid[0], mid[1] + 3, mid[2])
            follower.GetProperty().SetColor(*color)
            follower.SetScale(0.2, 0.2, 0.2)
            self.renderer.AddActor(follower)
            self._minmax_actors.append(follower)

    def clear_min_max_indicators(self):
        """清除最短/最长距离标示"""
        for actor in self._minmax_actors:
            self.renderer.RemoveActor(actor)
        self._minmax_actors.clear()

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
        """
        添加带刻度的坐标轴

        参数:
            bounds: 模型包围盒 (xMin, xMax, yMin, yMax, zMin, zMax)
        """
        # 移除旧的坐标轴
        if self.axes_actor is not None:
            self.renderer.RemoveActor(self.axes_actor)
        if hasattr(self, '_label_actors'):
            for actor in self._label_actors:
                self.renderer.RemoveActor(actor)
        if hasattr(self, '_tick_actors'):
            for actor in self._tick_actors:
                self.renderer.RemoveActor(actor)

        # 计算各轴范围
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]

        # 坐标轴长度 = 模型范围 * 1.2（比模型长 20%）
        axes_length = max(x_range, y_range, z_range) * 1.2
        self.axes_length = axes_length

        # 计算中心点
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ]

        # 创建简单的坐标轴线
        self._create_axis_line(center, axes_length, 'X')
        self._create_axis_line(center, axes_length, 'Y')
        self._create_axis_line(center, axes_length, 'Z')

        # 添加轴标签
        self._add_axis_labels(center, axes_length)

    def _create_axis_line(self, center: list, axes_length: float, axis: str):
        """
        创建单个坐标轴线

        参数:
            center: 坐标原点
            axes_length: 坐标轴长度
            axis: 'X', 'Y', 或 'Z'
        """
        half_length = axes_length / 2

        # 设置轴线起点和终点
        if axis == 'X':
            p1 = [center[0] - half_length, center[1], center[2]]
            p2 = [center[0] + half_length, center[1], center[2]]
            color = (1, 0, 0)  # 红色
        elif axis == 'Y':
            p1 = [center[0], center[1] - half_length, center[2]]
            p2 = [center[0], center[1] + half_length, center[2]]
            color = (0, 1, 0)  # 绿色
        else:  # Z
            p1 = [center[0], center[1], center[2] - half_length]
            p2 = [center[0], center[1], center[2] + half_length]
            color = (0, 0, 1)  # 蓝色

        # 创建线
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

        # 保存箭头演员以便后续清理
        if not hasattr(self, '_axis_line_actors'):
            self._axis_line_actors = []
        self._axis_line_actors.append(actor)

    def _add_axis_labels(self, center: list, axes_length: float):
        """
        添加自定义坐标轴标签

        参数:
            center: 坐标原点
            axes_length: 坐标轴长度
        """
        # 移除旧标签
        if hasattr(self, '_label_actors'):
            for actor in self._label_actors:
                self.renderer.RemoveActor(actor)

        self._label_actors = []

        # 创建 X 轴标签
        x_label = vtk.vtkVectorText()
        x_label.SetText("X")
        x_mapper = vtk.vtkPolyDataMapper()
        x_mapper.SetInputConnection(x_label.GetOutputPort())
        x_actor = vtk.vtkFollower()
        x_actor.SetMapper(x_mapper)
        x_actor.SetPosition(center[0] + axes_length/2 + 3, center[1], center[2])
        x_actor.GetProperty().SetColor(1, 0, 0)
        x_actor.SetScale(0.3, 0.3, 0.3)
        self._label_actors.append(x_actor)
        self.renderer.AddActor(x_actor)

        # 创建 Y 轴标签
        y_label = vtk.vtkVectorText()
        y_label.SetText("Y")
        y_mapper = vtk.vtkPolyDataMapper()
        y_mapper.SetInputConnection(y_label.GetOutputPort())
        y_actor = vtk.vtkFollower()
        y_actor.SetMapper(y_mapper)
        y_actor.SetPosition(center[0], center[1] + axes_length/2 + 3, center[2])
        y_actor.GetProperty().SetColor(0, 1, 0)
        y_actor.SetScale(0.3, 0.3, 0.3)
        self._label_actors.append(y_actor)
        self.renderer.AddActor(y_actor)

        # 创建 Z 轴标签
        z_label = vtk.vtkVectorText()
        z_label.SetText("Z")
        z_mapper = vtk.vtkPolyDataMapper()
        z_mapper.SetInputConnection(z_label.GetOutputPort())
        z_actor = vtk.vtkFollower()
        z_actor.SetMapper(z_mapper)
        z_actor.SetPosition(center[0], center[1], center[2] + axes_length/2 + 3)
        z_actor.GetProperty().SetColor(0, 0, 1)
        z_actor.SetScale(0.3, 0.3, 0.3)
        self._label_actors.append(z_actor)
        self.renderer.AddActor(z_actor)

    def _add_axis_ticks(self, bounds: Tuple[float, float, float, float, float, float], axes_length: float, x_range: float, y_range: float, z_range: float):
        """
        为坐标轴添加刻度（2mm 间隔）

        参数:
            bounds: 模型包围盒
            axes_length: 坐标轴长度
        """
        # 移除旧的刻度
        if hasattr(self, '_tick_actors'):
            for actor in self._tick_actors:
                self.renderer.RemoveActor(actor)

        self._tick_actors = []

        # 刻度间隔（2mm）
        tick_interval = 2.0

        # 计算中心点
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ]

        # 为每个轴创建刻度
        for axis_idx, (axis_name, axis_range, origin) in enumerate([
            ('X', x_range, [0, 1, 2]),
            ('Y', y_range, [1, 0, 2]),
            ('Z', z_range, [2, 0, 1])
        ]):
            range_val = [x_range, y_range, z_range][axis_idx]
            half_range = range_val / 2

            # 计算刻度位置
            tick_count = int(range_val / tick_interval) + 1

            for i in range(-tick_count, tick_count + 1):
                pos = i * tick_interval
                if abs(pos) > axes_length / 2:
                    continue

                # 创建刻度线
                tick_source = vtk.vtkLineSource()
                if axis_idx == 0:  # X 轴刻度
                    tick_source.SetPoint1(center[0] + pos, center[1] - axes_length/2, center[2] - axes_length/2)
                    tick_source.SetPoint2(center[0] + pos, center[1] - axes_length/2 - 5, center[2] - axes_length/2)
                elif axis_idx == 1:  # Y 轴刻度
                    tick_source.SetPoint1(center[0] - axes_length/2, center[1] + pos, center[2] - axes_length/2)
                    tick_source.SetPoint2(center[0] - axes_length/2, center[1] + pos, center[2] - axes_length/2 - 5)
                else:  # Z 轴刻度
                    tick_source.SetPoint1(center[0] - axes_length/2, center[1] - axes_length/2, center[2] + pos)
                    tick_source.SetPoint2(center[0] - axes_length/2 - 5, center[1] - axes_length/2, center[2] + pos)

                tick_mapper = vtk.vtkPolyDataMapper()
                tick_mapper.SetInputConnection(tick_source.GetOutputPort())

                tick_actor = vtk.vtkActor()
                tick_actor.SetMapper(tick_mapper)

                # 设置刻度颜色（与轴颜色一致）
                if axis_idx == 0:
                    tick_actor.GetProperty().SetColor(1, 0, 0)
                elif axis_idx == 1:
                    tick_actor.GetProperty().SetColor(0, 1, 0)
                else:
                    tick_actor.GetProperty().SetColor(0, 0, 1)

                tick_actor.GetProperty().SetLineWidth(1)
                self._tick_actors.append(tick_actor)
                self.renderer.AddActor(tick_actor)

    # ---- 内部方法 ----

    def _update_scalar_bar(self, thresholds: list):
        """更新标量条显示"""
        self.scalar_bar.VisibilityOn()
        self.scalar_bar.SetNumberOfLabels(min(len(thresholds), 10))
        self.scalar_bar.SetTitle("Distance (mm)")
