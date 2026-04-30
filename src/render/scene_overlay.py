"""VTK 场景叠加层 — 测量、标记、指示器、文字覆盖"""

from typing import Optional

import numpy as np
import vtk


class SceneOverlay:
    """VTK 场景叠加层：测量、标记、指示器、文字"""

    def __init__(self):
        self._help_text: Optional[vtk.vtkTextActor] = None
        self._coord_text: Optional[vtk.vtkTextActor] = None
        self._measure_actors: list = []
        self._measure_status_text: Optional[vtk.vtkTextActor] = None
        self._cursor_text: Optional[vtk.vtkTextActor] = None
        self._pick_marker_actor: Optional[vtk.vtkActor] = None
        self._minmax_actors: list = []
        self._deform_point_marker: Optional[vtk.vtkActor] = None

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

    # ---- 测量工具 ----

    def add_measurement(self, p1: np.ndarray, p2: np.ndarray, distance: float):
        """添加测量可视化 Actor"""
        actors = []

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
        """在场景中标示最短距离和最长距离点"""
        self.clear_min_max_indicators()

        for brace_pt, foot_pt, val, color, label_prefix in [
            (min_pos, min_foot_pos, min_val, (1.0, 0.0, 1.0), "MIN"),
            (max_pos, max_foot_pos, max_val, (0.0, 1.0, 0.0), "MAX"),
        ]:
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

    # ---- 变形选点标记 ----

    def show_deform_point_marker(self, position: np.ndarray, transform=None):
        """在变形选点位置显示黄色高亮球体"""
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(3.5)
        sphere.SetThetaResolution(24)
        sphere.SetPhiResolution(24)
        sphere.SetCenter(position[0], position[1], position[2])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        if self._deform_point_marker is None:
            self._deform_point_marker = vtk.vtkActor()
            self._deform_point_marker.SetMapper(mapper)
            self._deform_point_marker.GetProperty().SetColor(1.0, 0.85, 0.0)
            self.renderer.AddActor(self._deform_point_marker)
        else:
            self._deform_point_marker.SetMapper(mapper)
            self._deform_point_marker.SetVisibility(True)

    def set_deform_point_marker_transform(self, transform):
        """设置黄球标记的 UserTransform"""
        if self._deform_point_marker is None:
            return
        if transform is not None:
            self._deform_point_marker.SetUserTransform(transform)
        else:
            self._deform_point_marker.SetUserTransform(None)

    def hide_deform_point_marker(self):
        """隐藏变形选点标记"""
        if self._deform_point_marker is not None:
            self._deform_point_marker.SetVisibility(False)
