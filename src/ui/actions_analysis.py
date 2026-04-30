"""距离计算、最优位置、径向距离分析相关操作"""

import numpy as np
import vtk
from PyQt5.QtWidgets import QMessageBox

from src.core.distance_calculator import DistanceCalculator
from src.core.optimizer import BraceOptimizer
from src.core.radial_distance_calculator import RadialDistanceCalculator
from src.core.surface_picker import get_transformed_vertices


class ActionsAnalysisMixin:
    """Mixin: 距离计算、最优位置搜索、径向距离分析、统计报告"""

    def _compute_distance(self):
        try:
            if self.foot_model is None:
                QMessageBox.warning(self, "警告", "请先加载足部模型")
                return
            if self.brace_model is None:
                QMessageBox.warning(self, "警告", "请先加载护具模型")
                return
            if self.distance_calc is None:
                QMessageBox.warning(self, "警告", "足部模型未正确初始化")
                return
            if self.brace_model.polydata is None:
                QMessageBox.warning(self, "警告", "护具 polydata 无效")
                return

            self.status_bar.showMessage("正在计算距离...")
            self._render()

            selected_vertices = self._get_selected_inner_vertices()

            if selected_vertices is None or len(selected_vertices) == 0:
                if self._inner_vertex_indices is not None:
                    n_points = self.brace_model.polydata.GetNumberOfPoints()
                    valid_indices = np.array(
                        [i for i in self._inner_vertex_indices if 0 <= i < n_points],
                        dtype=np.int64,
                    )
                    if len(valid_indices) == 0:
                        self.status_bar.showMessage("计算失败：内侧面顶点索引全部无效，请重新选取内侧面")
                        QMessageBox.warning(
                            self, "警告",
                            "内侧面顶点索引已失效（可能因变形操作导致），\n请重新点击「选取内侧面」后再计算。"
                        )
                        return
                    selected_vertices = get_transformed_vertices(
                        self.brace_model.polydata,
                        valid_indices,
                        self.brace_transform.get_matrix(),
                    )
                    self._current_selected_indices = valid_indices
                else:
                    selected_vertices = self._polydata_to_vertices(self.brace_model.polydata)
                    selected_vertices = self.brace_transform.apply(selected_vertices)
                    self._current_selected_indices = None

            if selected_vertices is None or len(selected_vertices) == 0:
                self.status_bar.showMessage("计算失败：无有效顶点可供计算")
                QMessageBox.warning(self, "警告", "没有有效的顶点可供计算，请重新选取内侧面。")
                return

            stats = self.distance_calc.compute_with_stats(
                selected_vertices, self.current_thresholds
            )
            self.current_distances = stats.distances

            if self._current_selected_indices is not None:
                if len(self.current_distances) != len(self._current_selected_indices):
                    min_len = min(len(self.current_distances), len(self._current_selected_indices))
                    if min_len == 0:
                        self.status_bar.showMessage("计算失败：距离数据为空")
                        QMessageBox.warning(self, "警告", "距离计算结果为空，请检查模型状态。")
                        return
                    self._current_selected_indices = self._current_selected_indices[:min_len]
                    self.current_distances = self.current_distances[:min_len]

                self.scene.apply_inner_surface_colors(
                    self.brace_model.polydata,
                    self._current_selected_indices,
                    self.current_distances,
                    self.current_thresholds,
                )
            else:
                self.scene.apply_distance_colors(
                    self.brace_model.polydata,
                    self.current_distances,
                    self.current_thresholds,
                )

            self._last_calc_position = self.brace_transform.get_translation()
            self._update_stats(stats)
            self._draw_min_max_indicators(selected_vertices)
            self._render()

            self.status_bar.showMessage(
                f"计算完成 | 最小: {stats.min_dist:+.2f}mm | "
                f"平均: {stats.mean_dist:+.2f}mm | "
                f"穿透: {stats.penetration_count} 点"
            )

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[ComputeDistance Error] {e}\n{error_detail}")
            self.status_bar.showMessage("计算失败，详见错误弹窗")
            QMessageBox.critical(
                self, "计算失败",
                f"距离计算过程中发生错误：\n{e}\n\n"
                f"可能原因：\n"
                f"1. 护具模型经过变形操作后顶点索引失效\n"
                f"2. 内侧面数据已过期\n\n"
                f"建议：重新选取内侧面后再试。"
            )
            self.current_distances = None

    def _draw_min_max_indicators(self, selected_vertices: np.ndarray):
        if selected_vertices is None or len(selected_vertices) == 0:
            return
        if self.current_distances is None or len(self.current_distances) == 0:
            return
        if self.distance_calc is None:
            return

        try:
            if len(self.current_distances) > len(selected_vertices):
                self.current_distances = self.current_distances[:len(selected_vertices)]

            min_idx = int(np.argmin(self.current_distances))
            max_idx = int(np.argmax(self.current_distances))
            min_pos = selected_vertices[min_idx]
            max_pos = selected_vertices[max_idx]
            min_val = float(self.current_distances[min_idx])
            max_val = float(self.current_distances[max_idx])

            closest_point = np.zeros(3, dtype=np.float64)
            cell_id = vtk.mutable(0)
            sub_id = vtk.mutable(0)
            dist2 = vtk.mutable(0.0)

            self.distance_calc._locator.FindClosestPoint(
                min_pos, closest_point, cell_id, sub_id, dist2
            )
            min_foot_pos = closest_point.copy()

            self.distance_calc._locator.FindClosestPoint(
                max_pos, closest_point, cell_id, sub_id, dist2
            )
            max_foot_pos = closest_point.copy()

            self.scene.add_min_max_indicators(
                min_pos, min_val, max_pos, max_val,
                min_foot_pos, max_foot_pos,
            )
        except Exception as e:
            print(f"[DrawMinMax Error] {e}")

    def _optimize_position(self):
        if self.foot_model is None:
            QMessageBox.warning(self, "警告", "请先加载足部模型")
            return
        if self.brace_model is None:
            QMessageBox.warning(self, "警告", "请先加载护具模型")
            return
        if self._inner_vertex_indices is None or len(self._inner_vertex_indices) == 0:
            QMessageBox.warning(
                self, "警告",
                "请先选取内表面区域\n\n"
                "步骤：\n"
                "1. 点击'选取内侧面'按钮\n"
                "2. 在内侧面区域列表中勾选要计算的区域\n"
                "3. 再次点击'自动最优位置'"
            )
            return

        selected_vertices = self._get_selected_inner_vertices()
        if selected_vertices is None or len(selected_vertices) == 0:
            n_points = self.brace_model.polydata.GetNumberOfPoints()
            indices_to_use = np.array(
                [i for i in self._inner_vertex_indices if 0 <= i < n_points],
                dtype=np.int64,
            )
            if len(indices_to_use) == 0:
                self.status_bar.showMessage("计算失败：内侧面顶点索引无效，请重新选取内侧面")
                QMessageBox.warning(
                    self, "警告",
                    "内侧面顶点索引已失效，请重新点击「选取内侧面」后再计算。"
                )
                return
        else:
            indices_to_use = self._current_selected_indices

        self.status_bar.showMessage("正在计算最优位置（粗 + 精两级搜索）...")
        self._render()

        optimizer = BraceOptimizer(
            self.foot_model.polydata,
            self.brace_model.polydata,
            indices_to_use,
        )

        current_translation = self.brace_transform.get_translation()
        current_rotation = self.brace_transform.get_rotation()

        result = optimizer.optimize(
            current_translation=current_translation,
            current_rotation=current_rotation,
            search_range=10.0,
            search_step=2.0,
            rotation_range=5.0,
            rotation_step=2.5,
        )

        self.brace_transform.reset()
        self.brace_transform.translate(*result.translation.tolist())
        if np.any(result.rotation):
            self.brace_transform.rotate_x(result.rotation[0])
            self.brace_transform.rotate_y(result.rotation[1])
            self.brace_transform.rotate_z(result.rotation[2])

        self._update_brace_view()
        self._compute_distance()

        self._show_optimization_report(result)

        if result.min_distance < 0:
            QMessageBox.warning(
                self, "优化完成（有警告）",
                f"最优位置计算完成！\n\n"
                f"⚠ 最小径向间隙：{result.min_distance:+.2f} mm（存在穿透）\n\n"
                f"理想区间 (4-6mm) 覆盖率：{result.coverage_4_6mm:.1f}%\n"
                f"平均径向间隙：{result.mean_distance:.2f} mm\n\n"
                f"建议：手动调整护具远离足部后再运行优化。\n"
                f"详细结果已显示在'统计'标签页"
            )
        else:
            QMessageBox.information(
                self, "优化完成",
                f"最优位置计算完成！\n\n"
                f"最小径向间隙：{result.min_distance:+.2f} mm（无穿透）\n\n"
                f"理想区间 (4-6mm) 覆盖率：{result.coverage_4_6mm:.1f}%\n"
                f"平均径向间隙：{result.mean_distance:.2f} mm\n\n"
                f"详细结果已显示在'统计'标签页"
            )

    def _show_optimization_report(self, result):
        """显示优化结果统计报告（直接使用优化器返回的数据）"""
        report_text = (
            f"{'='*45}\n"
            f" 优化结果统计报告（径向距离）\n"
            f"{'='*45}\n\n"
            f"【优化位置】\n"
            f"  平移：X={result.translation[0]:+.2f}, "
            f"Y={result.translation[1]:+.2f}, "
            f"Z={result.translation[2]:+.2f} mm\n"
            f"  旋转：RX={result.rotation[0]:+.1f}°, "
            f"RY={result.rotation[1]:+.1f}°, "
            f"RZ={result.rotation[2]:+.1f}°\n\n"
            f"{'='*45}\n"
            f"【统计摘要】\n"
            f"  平均径向间隙：{result.mean_distance:.2f} mm\n"
            f"  标准差：{result.std_distance:.2f} mm\n"
            f"  最小间隙：{result.min_distance:+.2f} mm\n"
            f"  最大间隙：{result.max_distance:.2f} mm\n\n"
            f"{'='*45}\n"
            f"【理想区间分析 (4-6mm)】\n"
            f"  理想区间点数：{result.ideal_count} 点\n"
            f"  理想区间比例：{result.coverage_4_6mm:.1f}%\n"
            f"  总顶点数：{result.total_count}\n\n"
            f"{'='*45}\n"
        )

        self.stats_text.setText(report_text)

    def _compute_radial_distance(self):
        if self.foot_model is None:
            QMessageBox.warning(self, "警告", "请先加载足部模型")
            return
        if self.brace_model is None:
            QMessageBox.warning(self, "警告", "请先加载护具模型")
            return
        if self._inner_vertex_indices is None or len(self._inner_vertex_indices) == 0:
            QMessageBox.warning(
                self, "警告",
                "请先选取内侧面区域\n\n"
                "步骤：\n"
                "1. 点击'选取内侧面'按钮\n"
                "2. 在内侧面区域列表中勾选要计算的区域\n"
                "3. 再次点击'计算径向距离'"
            )
            return

        self.status_bar.showMessage("正在计算径向距离...")
        self._render()

        try:
            selected_vertices = self._get_selected_inner_vertices()

            if selected_vertices is None or len(selected_vertices) == 0:
                n_points = self.brace_model.polydata.GetNumberOfPoints()
                indices_to_use = np.array(
                    [i for i in self._inner_vertex_indices if 0 <= i < n_points],
                    dtype=np.int64,
                )
                if len(indices_to_use) == 0:
                    self.status_bar.showMessage("计算失败：内侧面顶点索引无效，请重新选取内侧面")
                    QMessageBox.warning(
                        self, "警告",
                        "内侧面顶点索引已失效，请重新点击「选取内侧面」后再计算。"
                    )
                    return
                selected_vertices = get_transformed_vertices(
                    self.brace_model.polydata,
                    indices_to_use,
                    self.brace_transform.get_matrix(),
                )
            else:
                indices_to_use = self._current_selected_indices

            transformed_brace = vtk.vtkTransformPolyDataFilter()
            transformed_brace.SetInputData(self.brace_model.polydata)
            transform = vtk.vtkTransform()
            transform.SetMatrix(self.brace_transform.get_matrix().flatten())
            transformed_brace.SetTransform(transform)
            transformed_brace.Update()

            self.radial_calc = RadialDistanceCalculator(
                foot_polydata=self.foot_model.polydata,
                brace_polydata=transformed_brace.GetOutput(),
                inner_vertex_indices=indices_to_use,
                axis_direction=np.array([0.0, 0.0, 1.0]),
            )

            stats = self.radial_calc.compute_radial_distances(selected_vertices)

            self._last_calc_position = self.brace_transform.get_translation()
            self.current_distances = stats.radial_distances
            self._current_selected_indices = indices_to_use

            self.scene.apply_inner_surface_colors(
                self.brace_model.polydata,
                self._current_selected_indices,
                self.current_distances,
                self.current_thresholds,
            )

            self._update_radial_stats(stats)
            self._draw_min_max_indicators(selected_vertices)
            self._render()

            self.status_bar.showMessage(
                f"径向距离计算完成 | 平均间隙：{stats.mean_gap:.2f}mm | "
                f"理想区间 (4-6mm): {stats.ideal_4_6_ratio:.1f}%"
            )

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[RadialDistance Error] {e}\n{error_detail}")
            self.status_bar.showMessage("径向距离计算失败，详见错误弹窗")
            QMessageBox.critical(
                self, "径向距离计算失败",
                f"计算过程中发生错误：\n{e}\n\n"
                f"可能原因：\n"
                f"1. 护具模型经过变形操作后顶点索引失效\n"
                f"2. 内侧面数据已过期\n\n"
                f"建议：重新选取内侧面后再试。"
            )

    def _update_radial_stats(self, stats):
        report_text = (
            f"{'='*45}\n"
            f" 径向距离分析报告（基于中轴线）\n"
            f"{'='*45}\n\n"
            f"【统计摘要】\n"
            f"  平均径向间隙：{stats.mean_gap:.2f} mm\n"
            f"  标准差：{stats.std_gap:.2f} mm\n"
            f"  最小间隙：{stats.min_gap:.2f} mm\n"
            f"  最大间隙：{stats.max_gap:.2f} mm\n\n"
            f"{'='*45}\n"
            f"【理想区间分析 (4-6mm)】\n"
            f"  理想区间点数：{stats.ideal_4_6_count} 点\n"
            f"  理想区间比例：{stats.ideal_4_6_ratio:.1f}%\n\n"
            f"{'='*45}\n"
        )

        report_text += "\n【距离分布】\n"
        bins = [0, 2, 4, 6, 8, 10, float('inf')]
        labels = ["0-2mm", "2-4mm", "4-6mm", "6-8mm", "8-10mm", ">10mm"]
        for i, (t_min, t_max) in enumerate(zip(bins[:-1], bins[1:])):
            if i == 0:
                count = int(np.sum((stats.radial_distances >= t_min) & (stats.radial_distances < t_max)))
            elif i == len(labels) - 1:
                count = int(np.sum(stats.radial_distances >= t_min))
            else:
                count = int(np.sum((stats.radial_distances >= t_min) & (stats.radial_distances < t_max)))
            ratio = count / len(stats.radial_distances) * 100 if len(stats.radial_distances) > 0 else 0
            bar = "█" * int(ratio / 5)
            report_text += f"  {labels[i]:>8}: {count:4} 点 ({ratio:5.1f}%) {bar}\n"

        self.stats_text.setText(report_text)

    def _update_stats(self, stats):
        distances = stats.distances
        total = len(distances)

        report_text = (
            f"{'='*45}\n"
            f" 有符号距离分析报告\n"
            f"{'='*45}\n\n"
            f"【统计摘要】\n"
            f"  平均距离：{stats.mean_dist:+.2f} mm\n"
            f"  中位数：{stats.median_dist:+.2f} mm\n"
            f"  标准差：{stats.std_dist:.2f} mm\n"
            f"  最小距离：{stats.min_dist:+.2f} mm\n"
            f"  最大距离：{stats.max_dist:+.2f} mm\n\n"
            f"{'='*45}\n"
            f"【理想区间分析 (4-6mm)】\n"
        )

        if total > 0:
            ideal = int(np.sum((distances >= 4) & (distances <= 6)))
            penetration = int(np.sum(distances < 0))
            report_text += (
                f"  理想区间点数：{ideal} 点\n"
                f"  理想区间比例：{ideal/total*100:.1f}%\n"
                f"  穿透点数：{penetration} 点\n\n"
            )
        else:
            report_text += "  无数据\n\n"

        report_text += f"{'='*45}\n"
        report_text += "\n【距离分布】\n"
        bins = [float('-inf'), 0, 2, 4, 6, 8, 10, float('inf')]
        labels = ["穿透(<0)", "0-2mm", "2-4mm", "4-6mm", "6-8mm", "8-10mm", ">10mm"]
        for i, (t_min, t_max) in enumerate(zip(bins[:-1], bins[1:])):
            if i == 0:
                count = int(np.sum(distances < t_max))
            elif i == len(labels) - 1:
                count = int(np.sum(distances >= t_min))
            else:
                count = int(np.sum((distances >= t_min) & (distances < t_max)))
            ratio = count / total * 100 if total > 0 else 0
            bar = "█" * int(ratio / 5)
            report_text += f"  {labels[i]:>8}: {count:4} 点 ({ratio:5.1f}%) {bar}\n"

        report_text += f"\n{'='*45}\n"
        report_text += f" 总顶点数：{total}\n"
        if total > 0:
            report_text += f" 理想覆盖率：{int(np.sum((distances >= 4) & (distances <= 6)))/total*100:.1f}%\n"
        report_text += f"{'='*45}"

        self.stats_text.setText(report_text)
