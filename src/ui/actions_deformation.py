"""变形引擎、选点、预览、撤销、重放、STL对比相关操作"""

import os
from typing import Optional

import numpy as np
import vtk
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QProgressDialog

from src.core.deformation_engine import DeformationEngine, DeformationParams
from src.core.deformation_state import DeformationState, DeformationStep


class _CompareWorker(QThread):
    """后台线程：执行 STL 对比计算（避免大文件阻塞主线程导致闪退）"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, filepath_a: str, filepath_b: str, tolerance: float):
        super().__init__()
        self.filepath_a = filepath_a
        self.filepath_b = filepath_b
        self.tolerance = tolerance

    def run(self):
        try:
            from src.core.stl_comparator import compare_stl
            report = compare_stl(
                self.filepath_a, self.filepath_b,
                tolerance=self.tolerance, same_topology=False
            )
            self.finished.emit(report)
        except Exception as e:
            self.error.emit(str(e))


class ActionsDeformationMixin:
    """Mixin: 变形模式、选点、预览、应用、撤销、重放、STL对比"""

    # ---- STL 对比 ----

    _compare_file_a: Optional[str] = None
    _compare_file_b: Optional[str] = None

    def _select_compare_file_a(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择 STL 文件 A", "", "STL Files (*.stl)"
        )
        if filepath:
            self._compare_file_a = filepath
            self.lbl_compare_a.setText(f"文件 A: {filepath}")

    def _select_compare_file_b(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择 STL 文件 B", "", "STL Files (*.stl)"
        )
        if filepath:
            self._compare_file_b = filepath
            self.lbl_compare_b.setText(f"文件 B: {filepath}")

    def _run_compare(self):
        if not self._compare_file_a:
            self.status_bar.showMessage("请先选择文件 A")
            return
        if not self._compare_file_b:
            self.status_bar.showMessage("请先选择文件 B")
            return

        self.status_bar.showMessage("正在对比 STL 文件...")
        self.compare_result.setText("计算中...\n\n大文件可能需要几分钟，请耐心等待。")

        tolerance = self.spin_compare_tol.value()
        self._compare_worker = _CompareWorker(
            self._compare_file_a, self._compare_file_b, tolerance
        )
        self._compare_worker.finished.connect(self._on_compare_done)
        self._compare_worker.error.connect(self._on_compare_error)
        self._compare_worker.start()

    def _on_compare_done(self, report):
        """后台对比完成，在主线程构建报告文本"""
        import os

        sep = "\u2500" * 60
        lines = []
        lines.append("=" * 60)
        lines.append("STL 文件结构对比报告")
        lines.append("=" * 60)

        s = report.stats_a
        label_a = os.path.basename(self._compare_file_a)
        lines.append(f"\n{label_a}:")
        lines.append(f"  顶点数:   {s.vertex_count:,}")
        lines.append(f"  三角面数: {s.triangle_count:,}")
        lines.append(f"  体积:     {s.volume:,.2f} mm\u00b3")
        lines.append(f"  表面积:   {s.surface_area:,.2f} mm\u00b2")
        lines.append(f"  质心:     ({s.centroid[0]:.2f}, {s.centroid[1]:.2f}, {s.centroid[2]:.2f})")
        lines.append(f"  尺寸:     {s.extent[0]:.2f} \u00d7 {s.extent[1]:.2f} \u00d7 {s.extent[2]:.2f} mm")

        s = report.stats_b
        label_b = os.path.basename(self._compare_file_b)
        lines.append(f"\n{label_b}:")
        lines.append(f"  顶点数:   {s.vertex_count:,}")
        lines.append(f"  三角面数: {s.triangle_count:,}")
        lines.append(f"  体积:     {s.volume:,.2f} mm\u00b3")
        lines.append(f"  表面积:   {s.surface_area:,.2f} mm\u00b2")
        lines.append(f"  质心:     ({s.centroid[0]:.2f}, {s.centroid[1]:.2f}, {s.centroid[2]:.2f})")
        lines.append(f"  尺寸:     {s.extent[0]:.2f} \u00d7 {s.extent[1]:.2f} \u00d7 {s.extent[2]:.2f} mm")

        lines.append(f"\n{sep}")
        lines.append("差异分析:")
        lines.append(f"  顶点差:   {report.stats_b.vertex_count - report.stats_a.vertex_count:+,}")
        lines.append(f"  面差:     {report.stats_b.triangle_count - report.stats_a.triangle_count:+,}")
        lines.append(f"  体积差:   {report.stats_b.volume - report.stats_a.volume:+,.2f} mm\u00b3")
        lines.append(f"  表面积差: {report.stats_b.surface_area - report.stats_a.surface_area:+,.2f} mm\u00b2")

        lines.append(f"\n{sep}")
        lines.append("拓扑对比:")
        topo_str = "是" if report.topology_match else "否"
        lines.append(f"  拓扑一致: {topo_str}")
        for diff in report.topology_differences[:5]:
            lines.append(f"  - {diff}")

        lines.append(f"\n{sep}")
        lines.append("几何距离 (mm) \u2014 A\u7684\u6bcf\u4e2a\u9876\u70b9\u5230B\u7684\u6700\u8fd1\u9876\u70b9\u8ddd\u79bb:")
        p10, p25, p50, p75, p90, p95, p99 = report.distance_percentiles
        lines.append(f"  平均:  {report.mean_distance:.4f}")
        lines.append(f"  RMS:   {report.rms_distance:.4f}")
        lines.append(f"  最大:  {report.max_distance:.4f}")
        lines.append(f"  P10:   {p10:.4f}  P25: {p25:.4f}  P50: {p50:.4f}")
        lines.append(f"  P75:   {p75:.4f}  P90: {p90:.4f}  P95: {p95:.4f}")
        lines.append(f"  P99:   {p99:.4f}")
        tolerance = self.spin_compare_tol.value()
        lines.append(f"  \u91cd\u53e0\u7387 (\u516c\u5dee {tolerance}mm \u5185): {report.overlap_ratio * 100:.1f}%")
        lines.append("=" * 60)

        # 文件大小对比
        size_a = os.path.getsize(self._compare_file_a)
        size_b = os.path.getsize(self._compare_file_b)
        lines.append(f"\n\u6587\u4ef6\u5927\u5c0f:")
        lines.append(f"  {label_a}: {size_a / 1024:.1f} KB")
        lines.append(f"  {label_b}: {size_b / 1024:.1f} KB")
        ratio = (size_b / size_a - 1) * 100 if size_a > 0 else 0
        lines.append(f"  \u5dee\u5f02:   {size_b - size_a:+,.0f} bytes ({ratio:+.1f}%)")
        lines.append("=" * 60)

        self.compare_result.setText("\n".join(lines))
        self.status_bar.showMessage(f"\u5bf9\u6bd4\u5b8c\u6210: {label_a} vs {label_b}")

    def _on_compare_error(self, error_msg: str):
        self.compare_result.setText(f"\u5bf9\u6bd4\u5931\u8d25:\n{error_msg}")
        self.status_bar.showMessage("STL \u5bf9\u6bd4\u5931\u8d25")

    # ---- 变形模式、选点、预览、应用、撤销、重放 ----

    def _on_deform_mode_changed(self):
        idx = self.combo_deform_mode.currentIndex()
        modes = ["normal", "normal_z", "directional", "radial", "adaptive"]
        self._deform_mode = modes[idx]

        # 方向选择 / 选点控件：仅 direction 模式可见
        self.dir_group.setVisible(self._deform_mode == "directional")
        self.point_group.setVisible(self._deform_mode == "directional")
        self.radial_group.setVisible(self._deform_mode == "radial")

        # 根据模式调整参数控件的标签和可见性
        self._update_deform_param_controls()

        # direction 模式特殊处理
        if self._deform_mode == "directional":
            self.lbl_deform_direction.setText("方向：选择拉伸轴向")
            self._ensure_deformation_engine()
            if self._deform_point_idx < 0:
                self._select_closest_to_foot()
        elif self._deform_mode == "normal":
            self.lbl_deform_direction.setText("方向：用偏移量正负号控制（正=向外扩张，负=向内收缩）— 质心放射")
        elif self._deform_mode == "normal_z":
            self.lbl_deform_direction.setText("方向：用偏移量正负号控制（正=向外扩张，负=向内收缩）— 沿Z轴圆柱")
        elif self._deform_mode == "radial":
            self.lbl_deform_direction.setText("方向：沿选中轴的径向缩放")
        elif self._deform_mode == "adaptive":
            self.lbl_deform_direction.setText("方向：自动计算偏移量，使平均间隙接近目标值")

        if self._deform_mode != "directional":
            self.scene.hide_deform_point_marker()
            self._render()

    def _update_deform_param_controls(self):
        """根据当前变形模式调整参数控件的显示"""
        mode = self._deform_mode

        if mode in ("normal", "normal_z"):
            # 法向变形：偏移量 (mm) + 衰减半径
            if mode == "normal":
                self.lbl_deform_offset.setText("偏移量 (mm): 正=向外扩张，负=向内收缩（质心放射）")
            else:
                self.lbl_deform_offset.setText("偏移量 (mm): 正=向外扩张，负=向内收缩（Z轴圆柱）")
            self.spin_deform_offset.setVisible(True)
            self.lbl_deform_offset.setVisible(True)
            self.lbl_deform_decay.setText("衰减半径 (mm, 0=均匀变形; ≥模型尺寸80%时生效，建议0=均匀):")
            self.spin_deform_decay.setVisible(True)
            self.lbl_deform_decay.setVisible(True)
            self.spin_deform_offset.setRange(-20.0, 20.0)
            self.spin_deform_offset.setSingleStep(0.5)
            self.spin_deform_offset.setValue(1.0)

        elif mode == "directional":
            # 方向拉伸：偏移量 (mm) + 衰减半径
            self.lbl_deform_offset.setText("偏移量 (mm): 正=向外扩张，负=向内收缩（选点驱动）")
            self.spin_deform_offset.setVisible(True)
            self.lbl_deform_offset.setVisible(True)
            self.lbl_deform_decay.setText("衰减半径 (mm, 0=均匀变形; ≥模型尺寸80%时生效，建议0=均匀):")
            self.spin_deform_decay.setVisible(True)
            self.lbl_deform_decay.setVisible(True)
            self.spin_deform_offset.setRange(-20.0, 20.0)
            self.spin_deform_offset.setSingleStep(0.5)
            self.spin_deform_offset.setValue(1.0)

        elif mode == "radial":
            # 径向缩放：缩放比例 (%) + 衰减半径
            self.lbl_deform_offset.setText("缩放比例 (%): 正=径向放大，负=径向缩小（沿选中轴）")
            self.spin_deform_offset.setVisible(True)
            self.lbl_deform_offset.setVisible(True)
            self.lbl_deform_decay.setText("衰减半径 (mm, 0=均匀缩放; ≥模型尺寸80%时生效，建议0=均匀):")
            self.spin_deform_decay.setVisible(True)
            self.lbl_deform_decay.setVisible(True)
            self.spin_deform_offset.setRange(-50.0, 50.0)
            self.spin_deform_offset.setSingleStep(1.0)
            self.spin_deform_offset.setValue(5.0)

        elif mode == "adaptive":
            # 自适应：目标间隙 (mm)，不需要衰减半径
            self.lbl_deform_offset.setText("目标间隙 (mm): 护具自动调整到该平均间隙")
            self.spin_deform_offset.setVisible(True)
            self.lbl_deform_offset.setVisible(True)
            self.spin_deform_decay.setVisible(False)
            self.lbl_deform_decay.setVisible(False)
            self.spin_deform_offset.setRange(0.5, 30.0)
            self.spin_deform_offset.setSingleStep(0.5)
            self.spin_deform_offset.setValue(5.0)

    def _get_custom_direction(self) -> Optional[np.ndarray]:
        idx = self._dir_custom.currentIndex()
        axes = [
            None,
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        ]
        return axes[idx]

    def _get_radial_axis(self) -> Optional[np.ndarray]:
        """获取径向模式选中的轴方向，None = PCA 中轴线"""
        idx = self._radial_axis.currentIndex()
        if idx == 0:
            return None
        axes = [
            None,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        return axes[idx]

    def _move_deform_point(self, direction: str):
        if self._deform_point_idx < 0 or self._deform_point_idx not in self._inner_adjacency:
            return

        neighbors = self._inner_adjacency[self._deform_point_idx]
        if not neighbors:
            return

        verts = self._base_inner_vertices if self._base_inner_vertices is not None else self._deformed_inner_vertices
        brace_transform = self.scene._brace_actor.GetUserTransform() if self.scene._brace_actor else None
        if brace_transform is not None:
            mtx = brace_transform.GetMatrix()
            transform_matrix = np.array([
                [mtx.GetElement(i, j) for j in range(4)] for i in range(4)
            ], dtype=np.float64)
            homogeneous = np.hstack([verts, np.ones((len(verts), 1))])
            world_verts = (homogeneous @ transform_matrix.T)[:, :3]
        else:
            world_verts = verts.copy()

        camera = self.scene.renderer.GetActiveCamera()
        pos = np.array(camera.GetPosition())
        fp = np.array(camera.GetFocalPoint())
        view_up = np.array(camera.GetViewUp())

        cam_z = (pos - fp)
        cam_z = cam_z / np.linalg.norm(cam_z)
        cam_x = np.cross(view_up, cam_z)
        cam_x = cam_x / np.linalg.norm(cam_x)
        cam_y = np.cross(cam_z, cam_x)

        current_world = world_verts[self._deform_point_idx]
        delta = world_verts[neighbors] - current_world

        screen_x = delta @ cam_x
        screen_y = delta @ cam_y

        screen_dir = {
            "up":    screen_y,
            "down":  -screen_y,
            "left":  -screen_x,
            "right": screen_x,
        }
        scores = screen_dir.get(direction, np.zeros_like(screen_x))
        best_idx = int(np.argmax(scores))
        target = neighbors[best_idx]

        self._deform_point_idx = target
        self._update_deform_point_marker()

        pos = world_verts[self._deform_point_idx]
        self.lbl_deform_point.setText(
            f"idx={self._deform_point_idx} ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
        )
        self.lbl_deform_point.setStyleSheet("color: #4a4;")
        self.status_bar.showMessage(
            f"选点移动到: idx={self._deform_point_idx}, "
            f"位置=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
        )

    def _update_deform_point_marker(self):
        if self._deform_point_idx < 0:
            self.scene.hide_deform_point_marker()
            return

        if self.brace_model is None or self.scene._brace_actor is None:
            return

        mesh_idx = int(self._inner_vertex_indices[self._deform_point_idx])

        mapper = self.scene._brace_actor.GetMapper()
        mapper_input = mapper.GetInput()
        if mapper_input is not None:
            display_pos = np.array(mapper_input.GetPoint(mesh_idx))
        else:
            pts = self.brace_model.polydata.GetPoints()
            display_pos = np.array(pts.GetPoint(mesh_idx))

        brace_transform = self.scene._brace_actor.GetUserTransform()
        if brace_transform is not None:
            world_pos = np.array(brace_transform.TransformPoint(display_pos))
        else:
            world_pos = display_pos

        self.scene.show_deform_point_marker(world_pos, None)
        self._render()

    def _set_deform_point(self, idx: int):
        if idx < 0 or idx >= len(self._inner_vertex_indices):
            return
        self._deform_point_idx = idx
        self._update_deform_point_marker()

    def _select_closest_to_foot(self):
        """自动选距足部最近的点作为变形中心

        优先使用已有距离计算中的最短距离点（与贴合分析中标注的最短距离点一致），
        如果没有距离计算结果则重新计算。
        """
        if self.foot_model is None:
            self.status_bar.showMessage("错误：请先加载足部模型")
            return

        if self.brace_model is None:
            self.status_bar.showMessage("错误：请先加载护具模型")
            return

        if self._inner_vertex_indices is None or len(self._inner_vertex_indices) == 0:
            self.status_bar.showMessage("错误：请先选取内侧面")
            return

        # 确保变形引擎已初始化（用于对比计算）
        self._ensure_deformation_engine()

        # 优先使用已有距离计算的最短距离点
        if self.current_distances is not None and len(self.current_distances) > 0:
            min_idx_in_distances = int(np.argmin(self.current_distances))
            # _current_selected_indices 是全局网格索引，映射到内表面数组索引
            if hasattr(self, '_current_selected_indices') and self._current_selected_indices is not None:
                global_idx = int(self._current_selected_indices[min_idx_in_distances])
                index_map = {idx: i for i, idx in enumerate(self._inner_vertex_indices)}
                if global_idx in index_map:
                    self._deform_point_idx = index_map[global_idx]
                    verts = self._deformed_inner_vertices if self._deformed_inner_vertices is not None else self._base_inner_vertices
                    pos = verts[self._deform_point_idx]

                    # 对比：如果不使用 current_distances，自动选点会找到哪个点
                    selected_indices = self._get_selected_region_indices()
                    if selected_indices is None or len(selected_indices) == 0:
                        selected_indices = np.arange(len(self._inner_vertex_indices), dtype=np.int64)

                    # 获取距离计算时用的顶点（世界坐标，含brace_transform）
                    from src.core.surface_picker import get_transformed_vertices
                    if len(self._current_selected_indices) > 0:
                        calc_verts = get_transformed_vertices(
                            self.brace_model.polydata,
                            self._current_selected_indices,
                            self.brace_transform.get_matrix(),
                        )
                    else:
                        calc_verts = verts  # fallback

                    # 自动选点用的顶点（局部坐标）
                    auto_verts = verts[selected_indices]

                    # 自动选点用的几何距离
                    if self.deformation_engine is not None:
                        foot_points = self.deformation_engine._find_closest_foot_points(auto_verts)
                        gaps = np.linalg.norm(auto_verts - foot_points, axis=1)

                    # 用两个不同的 locator 分别计算
                    foot_points_calc = self.distance_calc._locator.FindClosestPoint if hasattr(self, 'distance_calc') and self.distance_calc else None
                    print(f"[对比] 距离计算顶点数={len(calc_verts)}, 自动选点顶点数={len(auto_verts)}")
                    print(f"[对比] brace_transform = {self.brace_transform.get_matrix()}")
                    if foot_points_calc is not None:
                        # 距离计算的 MIN 点在 calc_verts 中的局部坐标
                        min_pos_calc = calc_verts[min_idx_in_distances]
                        print(f"[对比] 距离计算 MIN 世界坐标: ({min_pos_calc[0]:.1f}, {min_pos_calc[1]:.1f}, {min_pos_calc[2]:.1f})")
                        # 自动选点的 MIN 点在 auto_verts 中的局部坐标
                        auto_closest = int(np.argmin(gaps))
                        auto_pos = auto_verts[auto_closest]
                        print(f"[对比] 自动选点最近 局部坐标: ({auto_pos[0]:.1f}, {auto_pos[1]:.1f}, {auto_pos[2]:.1f})")
                        # 用距离计算用的 locator 重新计算自动选点顶点的最近距离
                        from src.core.distance_calculator import DistanceCalculator
                        dc = self.distance_calc
                        if dc is not None:
                            closest_point = np.zeros(3, dtype=np.float64)
                            cell_id = vtk.mutable(0)
                            sub_id = vtk.mutable(0)
                            dist2 = vtk.mutable(0.0)
                            dc._locator.FindClosestPoint(auto_pos, closest_point, cell_id, sub_id, dist2)
                            dist_via_dc = np.sqrt(float(dist2))
                            print(f"[对比] 用距离计算器的locator，auto_pos距足部: {dist_via_dc:.2f}mm")

                    print(f"[自动选点] 使用最短距离点: 内表面 idx={self._deform_point_idx}, 距离={self.current_distances[min_idx_in_distances]:+.2f}mm")
                    self.lbl_deform_point.setText(
                        f"idx={self._deform_point_idx} ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
                    )
                    self.lbl_deform_point.setStyleSheet("color: #4a4;")
                    self._update_deform_point_marker()
                    return

        # 没有距离计算结果，重新计算最近点
        self._ensure_deformation_engine()
        if self.deformation_engine is None:
            self.status_bar.showMessage("错误：变形引擎初始化失败")
            return

        selected_indices = self._get_selected_region_indices()
        if selected_indices is None or len(selected_indices) == 0:
            selected_indices = np.arange(len(self._inner_vertex_indices), dtype=np.int64)

        verts = self._deformed_inner_vertices if self._deformed_inner_vertices is not None else self._base_inner_vertices
        foot_points = self.deformation_engine._find_closest_foot_points(
            verts[selected_indices]
        )
        gaps = np.linalg.norm(
            verts[selected_indices] - foot_points, axis=1
        )
        region_closest = int(np.argmin(gaps))
        auto_idx = int(selected_indices[region_closest])

        # 对比：如果优先使用 current_distances 会得到什么结果
        if self.current_distances is not None and len(self.current_distances) > 0:
            min_idx_dist = int(np.argmin(self.current_distances))
            if hasattr(self, '_current_selected_indices') and self._current_selected_indices is not None:
                global_dist = int(self._current_selected_indices[min_idx_dist])
                index_map = {idx: i for i, idx in enumerate(self._inner_vertex_indices)}
                if global_dist in index_map:
                    dist_idx = index_map[global_dist]
                    print(f"[对比] 距离计算 MIN: 内表面 idx={dist_idx}, signed_dist={self.current_distances[min_idx_dist]:+.2f}mm")
                    print(f"[对比] 自动选点最近: 内表面 idx={auto_idx}, 几何距={gaps[region_closest]:.2f}mm")
                    if dist_idx != auto_idx:
                        print(f"  → 两个算法找到了不同的点！差值 idx={abs(dist_idx - auto_idx)}")
                        pos_dist = verts[dist_idx]
                        pos_auto = verts[auto_idx]
                        print(f"  → 距离计算位置: ({pos_dist[0]:.1f}, {pos_dist[1]:.1f}, {pos_dist[2]:.1f})")
                        print(f"  → 自动选点位置: ({pos_auto[0]:.1f}, {pos_auto[1]:.1f}, {pos_auto[2]:.1f})")

        self._deform_point_idx = auto_idx
        pos = verts[self._deform_point_idx]
        print(f"自动选点(重算): 内表面 idx={self._deform_point_idx}, 距足部 {gaps[region_closest]:.2f}mm, 位置=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        self.lbl_deform_point.setText(
            f"idx={self._deform_point_idx} ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
        )
        self.lbl_deform_point.setStyleSheet("color: #4a4;")
        self._update_deform_point_marker()

    def _ensure_deformation_engine(self):
        if (self.deformation_engine is None
                and self.foot_model is not None
                and self.brace_model is not None
                and self._inner_vertex_indices is not None):

            self.deformation_engine = DeformationEngine(
                self.foot_model.polydata,
                self.brace_model.polydata,
                self._inner_vertex_indices,
            )
            pts = self.brace_model.polydata.GetPoints()
            self._base_inner_vertices = np.array([
                pts.GetPoint(i) for i in self._inner_vertex_indices
            ])
            self._original_inner_vertices = self._base_inner_vertices.copy()
            self._deformed_inner_vertices = self._original_inner_vertices.copy()
            if self._deform_point_idx >= 0:
                pos = self._base_inner_vertices[self._deform_point_idx]
                self.lbl_deform_point.setText(
                    f"idx={self._deform_point_idx} ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
                )

    def _bake_brace_transform(self):
        if self.scene._brace_actor is None:
            return
        brace_transform = self.scene._brace_actor.GetUserTransform()
        if brace_transform is None:
            return

        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

        mtx = brace_transform.GetMatrix()
        transform_matrix = np.array([
            [mtx.GetElement(i, j) for j in range(4)] for i in range(4)
        ], dtype=np.float64)

        pts = self.brace_model.polydata.GetPoints()
        pts_array = vtk_to_numpy(pts.GetData())
        n = len(pts_array)
        homogeneous = np.hstack([pts_array, np.ones((n, 1))])
        transformed = (homogeneous @ transform_matrix.T)[:, :3]

        new_pts = vtk.vtkPoints()
        new_pts.SetData(numpy_to_vtk(transformed))
        self.brace_model.polydata.SetPoints(new_pts)
        self.brace_model.polydata.Modified()

        mapper = self.scene._brace_actor.GetMapper()
        if mapper is not None:
            mapper_input = mapper.GetInput()
            if mapper_input is not None and mapper_input != self.brace_model.polydata:
                mpts = mapper_input.GetPoints()
                if mpts is not None:
                    m_array = vtk_to_numpy(mpts.GetData())
                    m_homogeneous = np.hstack([m_array, np.ones((len(m_array), 1))])
                    m_transformed = (m_homogeneous @ transform_matrix.T)[:, :3]
                    m_new_pts = vtk.vtkPoints()
                    m_new_pts.SetData(numpy_to_vtk(m_transformed))
                    mapper_input.SetPoints(m_new_pts)
                    mapper_input.Modified()
                    mapper.Modified()

        self.scene._brace_actor.SetUserTransform(None)

    def _get_deformation_params(self) -> Optional[DeformationParams]:
        selected_vertex_indices = self._get_selected_region_indices()

        if selected_vertex_indices is None or len(selected_vertex_indices) == 0:
            QMessageBox.warning(
                self, "警告",
                "请先在「模型」标签页的「内侧面区域」列表中勾选要变形的区域"
            )
            return None

        offset = self.spin_deform_offset.value()
        center = None
        direction = None

        if self._deform_mode == "directional":
            direction = self._get_custom_direction()
            if self._deform_point_idx >= 0:
                verts = self._deformed_inner_vertices if self._deformed_inner_vertices is not None else self._base_inner_vertices
                center = verts[self._deform_point_idx]
        elif self._deform_mode == "radial":
            verts = self._deformed_inner_vertices if self._deformed_inner_vertices is not None else self._base_inner_vertices
            center = np.mean(verts, axis=0)
            # 径向轴方向：通过 _direction 传递
            direction = self._get_radial_axis()

        # 自适应模式：spinbox 值 = 目标间隙，不是偏移量
        if self._deform_mode == "adaptive":
            return DeformationParams(
                mode=self._deform_mode,
                region_indices=selected_vertex_indices,
                offset_mm=0.0,
                scale_factor=1.0,
                decay_radius=0.0,
                boundary_smooth=0.0,
                direction=direction,
                center_point=center,
                _target_gap=offset,
            )

        return DeformationParams(
            mode=self._deform_mode,
            region_indices=selected_vertex_indices,
            offset_mm=offset,
            scale_factor=1.0 + offset / 100.0 if self._deform_mode == "radial" else 1.0,
            decay_radius=self.spin_deform_decay.value(),
            boundary_smooth=0.0,
            direction=direction,
            center_point=center,
        )

    def _get_selected_region_indices(self) -> Optional[np.ndarray]:
        if not self._inner_regions or self._inner_vertex_indices is None:
            return None

        if self.brace_model is None or self.brace_model.polydata is None:
            return None

        n_cells = self.brace_model.polydata.GetNumberOfCells()
        selected_cells = set()
        for i in range(self.region_list.count()):
            item = self.region_list.item(i)
            if item.checkState() == Qt.Checked:
                region_idx, _ = item.data(Qt.UserRole)
                if region_idx < len(self._inner_regions):
                    for cell_id in self._inner_regions[region_idx]:
                        if 0 <= cell_id < n_cells:
                            selected_cells.add(cell_id)

        if not selected_cells:
            return None

        n_points = self.brace_model.polydata.GetNumberOfPoints()
        selected_vertex_set = set()
        for cell_id in selected_cells:
            try:
                cell = self.brace_model.polydata.GetCell(cell_id)
                if cell is None:
                    continue
                for j in range(cell.GetNumberOfPoints()):
                    vid = cell.GetPointId(j)
                    if 0 <= vid < n_points:
                        selected_vertex_set.add(vid)
            except Exception:
                continue

        if not selected_vertex_set:
            return None

        index_map = {idx: i for i, idx in enumerate(self._inner_vertex_indices)}
        mapped_indices = np.array(
            sorted(index_map[v] for v in selected_vertex_set if v in index_map),
            dtype=np.int64,
        )
        return mapped_indices if len(mapped_indices) > 0 else None

    def _preview_deformation(self):
        if self.brace_model is None or self._inner_vertex_indices is None:
            self.status_bar.showMessage("错误：请先加载护具并在「模型」标签页选取内侧面")
            QMessageBox.warning(self, "警告", "请先加载护具并在「模型」标签页选取内侧面")
            return

        self._ensure_deformation_engine()
        if self.deformation_engine is None:
            self.status_bar.showMessage("错误：变形引擎初始化失败")
            QMessageBox.warning(self, "警告", "变形引擎初始化失败")
            return

        params = self._get_deformation_params()
        if params is None:
            return

        self.status_bar.showMessage(
            f"正在预览: 偏移={params.offset_mm:+.1f}mm, 区域顶点数 {len(params.region_indices):,} ..."
        )

        all_vertices = self._get_all_brace_vertices()
        deformed = self.deformation_engine.apply(all_vertices, params)
        inner_idx = self._inner_vertex_indices.astype(int)
        self._deformed_inner_vertices = deformed[inner_idx].copy()
        self._preview_vertices = deformed
        self._apply_vertex_update_full(deformed)
        self._render()

        if self._deform_mode == "adaptive":
            self.status_bar.showMessage(
                f"预览中: 目标间隙 {params._target_gap:.1f}mm — "
                f"满意后点击「应用」提交，或点击「还原」取消"
            )
        else:
            direction_label = "向外扩张" if params.offset_mm > 0 else "向内收缩"
            self.status_bar.showMessage(
                f"预览中: {direction_label} {abs(params.offset_mm):.1f}mm — "
                f"满意后点击「应用」提交，或点击「还原」取消"
            )

    def _apply_deformation(self):
        if self.brace_model is None or self._inner_vertex_indices is None:
            QMessageBox.warning(self, "警告", "请先加载护具并在「模型」标签页选取内侧面")
            return

        self._ensure_deformation_engine()
        if self.deformation_engine is None:
            self.status_bar.showMessage("错误：变形引擎未初始化")
            return

        self.status_bar.showMessage(
            f"正在变形: 内表面 {len(self._inner_vertex_indices):,} 顶点 ..."
        )

        if self._preview_vertices is not None:
            final = self._preview_vertices
            params = self._get_deformation_params()
        else:
            params = self._get_deformation_params()
            if params is None:
                return
            all_vertices = self._get_all_brace_vertices()
            final = self.deformation_engine.apply(all_vertices, params)

        step = DeformationStep(
            mode=params.mode,
            offset_mm=params.offset_mm,
            scale_factor=params.scale_factor,
            decay_radius=params.decay_radius,
            boundary_smooth=params.boundary_smooth,
            direction=(
                params.direction.tolist() if params.direction is not None else None
            ),
            center_point=(
                params.center_point.tolist() if params.center_point is not None else None
            ),
            region_indices=params.region_indices.tolist(),
            target_gap=params._target_gap,
        )
        self.deformation_state.push(step)

        inner_idx = self._inner_vertex_indices.astype(int)
        self._deformed_inner_vertices = final.copy()[inner_idx]
        self._apply_vertex_update_full(final)

        self._preview_vertices = None
        self._render()

        if self.current_distances is not None:
            self._compute_distance()

        if self._deform_mode == "adaptive":
            self.status_bar.showMessage(
                f"已应用自适应: 目标间隙 {params._target_gap:.1f}mm "
                f"(共 {self.deformation_state.step_count} 步)"
            )
        else:
            direction_label = "向外扩张" if params.offset_mm > 0 else "向内收缩"
            n_regions = len(params.region_indices)
            self.status_bar.showMessage(
                f"已应用{direction_label} {abs(params.offset_mm):.1f}mm "
                f"(区域 {n_regions:,} 顶点, 共 {self.deformation_state.step_count} 步)"
            )

    def _revert_preview(self):
        """还原到原始模型（保留当前位移变换 + 内侧面选择）"""
        if self.brace_model is None:
            self.status_bar.showMessage("当前没有护具模型")
            return
        if self._original_brace_polydata is None:
            self.status_bar.showMessage("原始模型数据不存在，请重新加载护具")
            return

        # 保留当前位移变换
        current_translation = self.brace_transform.get_translation().copy()
        current_rotation = [
            self.brace_transform.get_rotation()[i]
            for i in range(3)
        ]

        # 保存内侧面选择状态
        saved_inner_cells = self._inner_cells
        saved_inner_vertex_indices = self._inner_vertex_indices
        saved_inner_regions = list(self._inner_regions)
        saved_region_checks = []
        for i in range(self.region_list.count()):
            item = self.region_list.item(i)
            checked = item.checkState() == Qt.Checked
            data = item.data(Qt.UserRole)
            saved_region_checks.append((checked, data))

        # 移除所有内侧面高亮 actor
        self.scene.clear_inner_surface_actors()
        self._region_actor_map.clear()

        # 用原始副本替换护具 polydata
        self.brace_model.polydata.DeepCopy(self._original_brace_polydata)
        self.brace_model.polydata.Modified()

        # 重建 mapper（恢复原始颜色）
        self._rebuild_brace_mapper()

        # 恢复位移变换
        self.brace_transform.reset()
        self.brace_transform.translate(*current_translation)
        self.brace_transform.rotate_x(current_rotation[0])
        self.brace_transform.rotate_y(current_rotation[1])
        self.brace_transform.rotate_z(current_rotation[2])

        # 重置变形状态
        self._preview_vertices = None
        self._deformed_inner_vertices = None
        self._original_inner_vertices = None
        self._base_inner_vertices = None
        self.deformation_engine = None
        self.deformation_state.clear()
        self.current_distances = None
        self.scene.clear_distance_colors()

        # 恢复内侧面选择
        self._inner_cells = saved_inner_cells
        self._inner_vertex_indices = saved_inner_vertex_indices
        self._inner_regions = saved_inner_regions

        # 恢复区域列表勾选状态
        self.region_list.clear()
        for i, (checked, data) in enumerate(saved_region_checks):
            from PyQt5.QtWidgets import QListWidgetItem
            item = QListWidgetItem()
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setData(Qt.UserRole, data)
            label = data[0] if data else f"区域 {i}"
            item.setText(str(label))
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.region_list.addItem(item)

        self._update_brace_view()
        self._render()
        self.status_bar.showMessage("已还原到原始模型（位移变换和内侧面选择保持不变）")

    def _undo_deformation(self):
        if self.deformation_state.step_count == 0:
            self.status_bar.showMessage("没有可撤销的操作")
            return

        self.deformation_state.undo()
        self._replay_deformations()
        self._render()
        self.status_bar.showMessage(
            f"已撤销 (剩余 {self.deformation_state.step_count} 步)"
        )

    def _replay_deformations(self):
        if self._base_inner_vertices is None:
            return
        if self.deformation_engine is None:
            return
        if self._original_brace_polydata is None:
            return

        # 先从原始模型重新初始化内表面顶点
        from vtk.util.numpy_support import vtk_to_numpy
        orig_vertices = vtk_to_numpy(self._original_brace_polydata.GetPoints().GetData())

        self._base_inner_vertices = orig_vertices[self._inner_vertex_indices.astype(int)].copy()
        self._deformed_inner_vertices = self._base_inner_vertices.copy()

        # 先把护具恢复到原始状态
        self.brace_model.polydata.DeepCopy(self._original_brace_polydata)
        self._rebuild_brace_mapper()

        # 重新初始化变形引擎（基于原始顶点）
        self.deformation_engine = DeformationEngine(
            self.foot_model.polydata,
            self.brace_model.polydata,
            self._inner_vertex_indices,
        )

        all_vertices = self._get_all_brace_vertices()

        for step in self.deformation_state.get_params_list():
            deformed_all = self.deformation_engine.apply(all_vertices, step)
            self._deformed_inner_vertices = deformed_all[self._inner_vertex_indices.astype(int)].copy()
            self._apply_vertex_update_full(deformed_all)
            all_vertices = deformed_all

    def _apply_vertex_update(self, deformed_vertices: np.ndarray):
        if self.brace_model is None:
            return

        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

        inner_indices = self._inner_vertex_indices.astype(int)

        pts = self.brace_model.polydata.GetPoints()
        pts_array = vtk_to_numpy(pts.GetData())
        pts_array[inner_indices, :] = deformed_vertices
        pts.GetData().Modified()
        pts.Modified()
        self.brace_model.polydata.Modified()

        if self.scene._brace_actor is not None:
            mapper = self.scene._brace_actor.GetMapper()
            if mapper is not None:
                mapper_input = mapper.GetInput()
                if mapper_input is not None and mapper_input != self.brace_model.polydata:
                    mpts = mapper_input.GetPoints()
                    if mpts is not None and mpts.GetNumberOfPoints() == pts.GetNumberOfPoints():
                        m_array = vtk_to_numpy(mpts.GetData())
                        m_array[inner_indices, :] = deformed_vertices
                        mpts.GetData().Modified()
                        mpts.Modified()
                        mapper_input.Modified()
                        mapper.Modified()
                        self.scene._brace_actor.Modified()

    def _rebuild_brace_mapper(self):
        if self.scene._brace_actor is None:
            return

        actor = self.scene._brace_actor
        old_mapper = actor.GetMapper()

        scalar_range = None
        lut = None
        has_scalars = False
        if old_mapper is not None:
            scalar_range = old_mapper.GetScalarRange()
            lut = old_mapper.GetLookupTable()
            old_input = old_mapper.GetInput()
            if old_input is not None and old_input.GetPointData() is not None:
                if old_input.GetPointData().GetScalars() is not None:
                    has_scalars = True

        new_mapper = vtk.vtkPolyDataMapper()
        new_mapper.SetInputData(self.brace_model.polydata)

        if has_scalars and scalar_range is not None:
            new_mapper.SetScalarModeToUsePointData()
            new_mapper.SetScalarRange(scalar_range[0], scalar_range[1])
            if lut is not None:
                new_mapper.SetLookupTable(lut)
        else:
            from src.config import BRACE_COLOR, MODEL_OPACITY_NORMAL
            actor.GetProperty().SetColor(*BRACE_COLOR)
            actor.GetProperty().SetOpacity(MODEL_OPACITY_NORMAL)

        actor.SetMapper(new_mapper)

    def _get_all_brace_vertices(self) -> np.ndarray:
        from vtk.util.numpy_support import vtk_to_numpy
        return vtk_to_numpy(self.brace_model.polydata.GetPoints().GetData())

    def _apply_vertex_update_full(self, all_deformed_vertices: np.ndarray):
        if self.brace_model is None:
            return

        from vtk.util.numpy_support import vtk_to_numpy

        pts = self.brace_model.polydata.GetPoints()
        pts_array = vtk_to_numpy(pts.GetData())
        pts_array[:, :] = all_deformed_vertices
        pts.GetData().Modified()
        pts.Modified()
        self.brace_model.polydata.Modified()

        if self.scene._brace_actor is not None:
            mapper = self.scene._brace_actor.GetMapper()
            if mapper is not None:
                mapper_input = mapper.GetInput()
                if mapper_input is not None and mapper_input != self.brace_model.polydata:
                    mpts = mapper_input.GetPoints()
                    if mpts is not None and mpts.GetNumberOfPoints() == pts.GetNumberOfPoints():
                        m_array = vtk_to_numpy(mpts.GetData())
                        m_array[:, :] = all_deformed_vertices
                        mpts.GetData().Modified()
                        mpts.Modified()
                        mapper_input.Modified()
                        mapper.Modified()
                        self.scene._brace_actor.Modified()
