"""导出、截图、测量、颜色、线框、设置等工具操作"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import vtk
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QTableWidgetItem,
)

from src.config import PRESET_THRESHOLDS
from src.core.distance_measurement import DistanceMeasurement
from src.core.step_converter import stl_to_step


class ActionsToolsMixin:
    """Mixin: 导出、截图、测量、颜色阈值、线框、显示设置、工作区管理"""

    def _export_stl(self):
        if self.brace_model is None:
            QMessageBox.warning(self, "警告", "请先加载护具")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出 STL", "", "STL Files (*.stl)"
        )
        if not filepath:
            return

        try:
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(filepath)
            writer.SetInputData(self.brace_model.polydata)
            writer.Write()
            self.status_bar.showMessage(f"已导出 STL: {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"无法导出 STL:\n{e}")

    def _export_step(self):
        if self.brace_model is None:
            QMessageBox.warning(self, "警告", "请先加载护具")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出 STEP", "", "STEP Files (*.step *.stp)"
        )
        if not filepath:
            return

        try:
            temp_stl = filepath + ".temp.stl"
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(temp_stl)
            writer.SetInputData(self.brace_model.polydata)
            writer.Write()

            stl_to_step(temp_stl, filepath, tolerance=0.1)

            if os.path.exists(temp_stl):
                os.unlink(temp_stl)

            self.status_bar.showMessage(f"已导出 STEP: {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"无法导出 STEP:\n{e}")

    def _screenshot(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存截图", "", "PNG Images (*.png)"
        )
        if not filepath:
            return
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.render_window)
        w2i.SetScale(2)
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filepath)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        self.status_bar.showMessage(f"截图已保存: {filepath}")

    def _toggle_wireframe(self):
        self._wireframe = not self._wireframe
        if self.scene._foot_actor:
            self.scene.set_wireframe(
                self.scene._foot_actor, self._wireframe
            )
        if self.scene._brace_actor:
            self.scene.set_wireframe(
                self.scene._brace_actor, self._wireframe
            )
        self._render()

    def _toggle_measurement(self, checked: bool):
        self._measure_tool_active = checked
        self.custom_style.set_measuring(checked)
        self.btn_measure.setChecked(checked)
        if hasattr(self, "measure_action"):
            self.measure_action.setChecked(checked)

        if checked:
            self.scene.clear_measurements()
            self._measure_calc.reset()
            self.scene.add_measurement_status_text("点击模型上的第一个点")
            self.status_bar.showMessage("测量模式: 点击模型上的第一个点")
        else:
            self.scene.hide_measurement_status_text()
            self.scene.hide_cursor_text()
            self.status_bar.showMessage("已退出测量模式，测量结果已保留在场景中")

        self._render()

    def _on_measurement_click(self, display_x: int, display_y: int):
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(display_x, display_y, 0, self.scene.renderer)

        actor = picker.GetActor()
        if actor is not None and actor is not self.scene._foot_actor and actor is not self.scene._brace_actor:
            actor = None

        if actor is None:
            self.status_bar.showMessage("未点到模型，请重新点击")
            return

        world_pos = np.array(picker.GetPickPosition(), dtype=np.float64)

        self._measure_calc.add_point(world_pos)

        if self._measure_calc.point_count == 1:
            self.scene.add_pick_marker(world_pos)
            self.scene.add_measurement_status_text(
                f"第1点: ({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f}) — 点击第二个点"
            )
            self.status_bar.showMessage(
                f"已选第1点: ({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f})"
            )

        elif self._measure_calc.is_complete:
            result = self._measure_calc.get_result()
            if result is not None:
                self.scene.add_measurement(
                    result.point1, result.point2, result.distance
                )
                self.scene.clear_pick_marker()
                self.scene.add_measurement_status_text(
                    f"距离: {result.distance:.2f} mm  |  点击开始新测量"
                )
                self.status_bar.showMessage(
                    f"P1({result.point1[0]:.2f}, {result.point1[1]:.2f}, {result.point1[2]:.2f})  "
                    f"P2({result.point2[0]:.2f}, {result.point2[1]:.2f}, {result.point2[2]:.2f})  "
                    f"距离: {result.distance:.2f} mm"
                )
                self._measure_calc.reset()

        self._render()

    def _on_measurement_move(self, display_x: int, display_y: int):
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(display_x, display_y, 0, self.scene.renderer)

        actor = picker.GetActor()
        if actor is not None and actor is not self.scene._foot_actor and actor is not self.scene._brace_actor:
            actor = None

        if actor is None:
            self.scene.hide_cursor_text()
        else:
            world_pos = np.array(picker.GetPickPosition(), dtype=np.float64)
            self.scene.show_cursor_text(world_pos)

        self._render()

    def _on_preset_changed(self, preset_name: str):
        if preset_name in PRESET_THRESHOLDS:
            self.current_thresholds = PRESET_THRESHOLDS[preset_name]
            self._refresh_threshold_table()
            if self.current_distances is not None:
                self.scene.apply_distance_colors(
                    self.brace_model.polydata,
                    self.current_distances,
                    self.current_thresholds,
                )
                self._render()

    def _refresh_threshold_table(self):
        self.threshold_table.setRowCount(len(self.current_thresholds))
        for i, (t_min, t_max, color, label) in enumerate(
            self.current_thresholds
        ):
            self.threshold_table.setItem(
                i, 0,
                QTableWidgetItem(str(t_min) if t_min != float("-inf") else "-∞")
            )
            self.threshold_table.setItem(
                i, 1,
                QTableWidgetItem(str(t_max) if t_max != float("inf") else "∞")
            )
            color_item = QTableWidgetItem(label)
            r, g, b = color
            qt_color = QColor(int(r * 255), int(g * 255), int(b * 255))
            color_item.setBackground(qt_color)
            color_item.setForeground(
                QColor(255, 255, 255) if (r + g + b) < 1.5 else QColor(0, 0, 0)
            )
            self.threshold_table.setItem(i, 2, color_item)
            self.threshold_table.setItem(i, 3, QTableWidgetItem(label))

    def _toggle_foot_visibility(self, checked: bool):
        if checked:
            self.btn_foot_visible.setText("隐藏足部")
            self.scene.set_foot_opacity(1.0)
        else:
            self.btn_foot_visible.setText("显示足部")
            self.scene.set_foot_opacity(0.0)
        self._render()

    def _set_foot_transparent(self, checked: bool):
        if checked:
            self.scene.set_foot_opacity(0.25)
        else:
            self.scene.set_foot_opacity(1.0)
        self._render()

    def _toggle_fine_mode(self, checked: bool):
        from src.config import DEFAULT_MOVE_STEP, DEFAULT_ROTATE_STEP
        self.is_fine_mode = checked
        if checked:
            self.move_step = 0.1
            self.rotate_step = 0.1
            self.spin_move.setValue(0.1)
            self.spin_rotate.setValue(0.1)
            self.btn_fine.setText("退出精细调节模式")
        else:
            self.move_step = DEFAULT_MOVE_STEP
            self.rotate_step = DEFAULT_ROTATE_STEP
            self.spin_move.setValue(DEFAULT_MOVE_STEP)
            self.spin_rotate.setValue(DEFAULT_ROTATE_STEP)
            self.btn_fine.setText("进入精细调节模式")

    def _on_move_step_changed(self, value: float):
        self.move_step = value
        if self.is_fine_mode:
            self.move_step = max(value, 0.01)

    def _on_rotate_step_changed(self, value: float):
        self.rotate_step = value

    # ---- 工作区管理 ----

    def _reset_transform(self):
        self.brace_transform.reset()
        self._update_brace_view()
        self.scene.clear_distance_colors()
        self.scene.clear_min_max_indicators()
        self.current_distances = None
        self.stats_text.clear()

        self.deformation_state.clear()
        self._deformed_inner_vertices = None
        self._original_inner_vertices = None
        self._base_inner_vertices = None

        self._refresh_inner_highlight()
        self._inner_cells = None
        self._inner_regions = []
        self._inner_vertex_indices = None
        self._region_actor_map.clear()
        self.region_list.clear()

        self._render()

    def _clear_workspace(self):
        try:
            self._refresh_inner_highlight()
        except Exception:
            pass

        self._safe_remove_actor(self.scene._foot_actor)
        self.scene._foot_actor = None
        self._safe_remove_actor(self.scene._brace_actor)
        self.scene._brace_actor = None

        try:
            self.scene.clear_distance_colors()
            self.scene.clear_min_max_indicators()
            self.scene.hide_deform_point_marker()
        except Exception:
            pass

        self._safe_remove_actor(self.scene.axes_actor)
        self.scene.axes_actor = None

        for attr in ('_axis_line_actors', '_label_actors', '_tick_actors'):
            if hasattr(self.scene, attr):
                actors = getattr(self.scene, attr)
                for actor in (actors or []):
                    self._safe_remove_actor(actor)
                setattr(self.scene, attr, [])

        self._safe_remove_actor(getattr(self.scene, '_coord_text', None))
        self.scene._coord_text = None

        from src.core.transform_manager import TransformManager
        self.foot_model = None
        self.brace_model = None
        self.brace_transform = TransformManager()
        self.distance_calc = None
        self.radial_calc = None
        self.current_distances = None
        self._last_calc_position = np.zeros(3)
        self._pending_recalc = False
        self.deformation_state.clear()
        self.deformation_engine = None

        self._inner_cells = None
        self._inner_regions = []
        self._inner_vertex_indices = None
        self._region_actor_map.clear()
        self._inner_adjacency = {}
        self._base_inner_vertices = None
        self._original_inner_vertices = None
        self._deformed_inner_vertices = None
        self._preview_vertices = None
        self._deform_point_idx = -1

        self.saved_positions = [None] * 5
        self._refresh_position_list()
        self._current_brace_filepath = None
        self._brace_step_filepath = None

        self.lbl_foot.setText("足部模型: 未加载")
        self.lbl_brace.setText("护具模型: 未加载")
        self.lbl_vertices.setText("顶点数: -")
        self.stats_text.clear()
        self.region_list.clear()

        self._update_coord_display()

        self.scene.reset_camera()
        self._render()

        self.status_bar.showMessage("工作区已清除，请加载新的足部和护具模型")

    # ---- 视图更新 ----

    def _reset_camera(self):
        self.scene.reset_camera()
        self._render()

    def _update_brace_view(self):
        if self.scene._brace_actor is None:
            return
        self.scene.apply_transform(self.brace_transform.get_matrix())
        self._update_coord_display()

        if (
            self.distance_calc is not None
            and self.current_distances is not None
        ):
            current_pos = self.brace_transform.get_translation()
            displacement = np.linalg.norm(
                current_pos - self._last_calc_position
            )
            if displacement >= 0.5:
                if not self._pending_recalc:
                    self._pending_recalc = True
                    from PyQt5.QtCore import QTimer
                    QTimer.singleShot(
                        200, self._debounced_recompute
                    )

        self._render()

    def _debounced_recompute(self):
        self._pending_recalc = False
        self._compute_distance()

    def _update_coord_display(self):
        t = self.brace_transform.get_translation()
        r = self.brace_transform.get_rotation()
        self.lbl_dx.setText(f"ΔX = {t[0]:+.2f} mm")
        self.lbl_dy.setText(f"ΔY = {t[1]:+.2f} mm")
        self.lbl_dz.setText(f"ΔZ = {t[2]:+.2f} mm")
        self.lbl_rx.setText(f"Rx = {r[0]:+.1f}°")
        self.lbl_ry.setText(f"Ry = {r[1]:+.1f}°")
        self.lbl_rz.setText(f"Rz = {r[2]:+.1f}°")

        coord_text = (
            f"护具位置：ΔX={t[0]:+.2f}  ΔY={t[1]:+.2f}  ΔZ={t[2]:+.2f} mm\n"
            f"旋转角度：Rx={r[0]:+.1f}° Ry={r[1]:+.1f}° Rz={r[2]:+.1f}°"
        )
        self.scene.add_coord_text(coord_text)

    def _update_axes(self):
        foot_bounds = self.foot_model.polydata.GetBounds()
        brace_bounds = self.brace_model.polydata.GetBounds()

        combined_bounds = (
            min(foot_bounds[0], brace_bounds[0]),
            max(foot_bounds[1], brace_bounds[1]),
            min(foot_bounds[2], brace_bounds[2]),
            max(foot_bounds[3], brace_bounds[3]),
            min(foot_bounds[4], brace_bounds[4]),
            max(foot_bounds[5], brace_bounds[5]),
        )

        self.scene.add_axes(combined_bounds)

    def _update_model_info(self):
        parts = []
        if self.foot_model:
            parts.append(f"足部: {self.foot_model.vertex_count:,} 顶点")
        if self.brace_model:
            parts.append(f"护具: {self.brace_model.vertex_count:,} 顶点")
        if parts:
            self.lbl_vertices.setText(" | ".join(parts))

    def _undo_transform(self):
        if self.brace_transform.undo():
            self._update_brace_view()
            self._render()
            self.status_bar.showMessage("已撤销")
