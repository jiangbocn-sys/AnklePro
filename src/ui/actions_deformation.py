"""变形引擎、选点、预览、撤销、重放相关操作"""

from typing import Optional

import numpy as np
import vtk
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from src.core.deformation_engine import DeformationEngine, DeformationParams
from src.core.deformation_state import DeformationState, DeformationStep


class ActionsDeformationMixin:
    """Mixin: 变形模式、选点、预览、应用、撤销、重放"""

    def _on_deform_mode_changed(self):
        idx = self.combo_deform_mode.currentIndex()
        modes = ["normal", "directional", "radial", "adaptive"]
        self._deform_mode = modes[idx]

        # 方向选择 / 选点控件：仅 direction 模式可见
        self.dir_group.setVisible(self._deform_mode == "directional")
        self.point_group.setVisible(self._deform_mode == "directional")

        # 根据模式调整参数控件的标签和可见性
        self._update_deform_param_controls()

        # direction 模式特殊处理
        if self._deform_mode == "directional":
            self.lbl_deform_direction.setText("方向：选择拉伸轴向")
            self._ensure_deformation_engine()
            if self._deform_point_idx < 0:
                self._select_closest_to_foot()
        elif self._deform_mode == "normal":
            self.lbl_deform_direction.setText("方向：用偏移量正负号控制（正=向外扩张，负=向内收缩）")
        elif self._deform_mode == "radial":
            self.lbl_deform_direction.setText("方向：绕中轴线径向缩放")
        elif self._deform_mode == "adaptive":
            self.lbl_deform_direction.setText("方向：自动计算偏移量，使平均间隙接近目标值")

        if self._deform_mode != "directional":
            self.scene.hide_deform_point_marker()
            self._render()

    def _update_deform_param_controls(self):
        """根据当前变形模式调整参数控件的显示"""
        mode = self._deform_mode

        if mode == "normal":
            # 法向变形：偏移量 (mm) + 衰减半径
            self.lbl_deform_offset.setText("偏移量 (mm):  正数=向外扩张，负数=向内收缩")
            self.spin_deform_offset.setVisible(True)
            self.lbl_deform_offset.setVisible(True)
            self.lbl_deform_decay.setText("衰减半径 (mm, 0=整个选区均匀变形):")
            self.spin_deform_decay.setVisible(True)
            self.lbl_deform_decay.setVisible(True)
            self.spin_deform_offset.setRange(-20.0, 20.0)
            self.spin_deform_offset.setSingleStep(0.5)
            self.spin_deform_offset.setValue(1.0)

        elif mode == "directional":
            # 方向拉伸：偏移量 (mm) + 衰减半径
            self.lbl_deform_offset.setText("偏移量 (mm):  正数=向外扩张，负数=向内收缩")
            self.spin_deform_offset.setVisible(True)
            self.lbl_deform_offset.setVisible(True)
            self.lbl_deform_decay.setText("衰减半径 (mm, 0=整个选区均匀变形):")
            self.spin_deform_decay.setVisible(True)
            self.lbl_deform_decay.setVisible(True)
            self.spin_deform_offset.setRange(-20.0, 20.0)
            self.spin_deform_offset.setSingleStep(0.5)
            self.spin_deform_offset.setValue(1.0)

        elif mode == "radial":
            # 径向缩放：缩放比例 (%) + 衰减半径
            self.lbl_deform_offset.setText("缩放比例 (%):  正数=放大，负数=缩小")
            self.spin_deform_offset.setVisible(True)
            self.lbl_deform_offset.setVisible(True)
            self.lbl_deform_decay.setText("衰减半径 (mm, 0=整个选区均匀缩放):")
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
        if self.deformation_engine is None or self._inner_vertex_indices is None:
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
        self._deform_point_idx = int(selected_indices[region_closest])

        print(f"自动选点: 内表面 idx={self._deform_point_idx}, 距足部 {gaps[region_closest]:.2f}mm")
        pos = verts[self._deform_point_idx]
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

        if params.mode == "directional":
            all_vertices = self._get_all_brace_vertices()
            deformed = self.deformation_engine.apply(all_vertices, params)
            inner_idx = self._inner_vertex_indices.astype(int)
            all_idx = set(inner_idx.tolist())
            outer_idx = [i for i in range(len(all_vertices)) if i not in all_idx]
            inner_disp = np.linalg.norm(deformed[inner_idx] - all_vertices[inner_idx], axis=1)
            if outer_idx:
                outer_idx = np.array(outer_idx)
                outer_disp = np.linalg.norm(deformed[outer_idx] - all_vertices[outer_idx], axis=1)
                print(f"[方向拉伸] 总顶点 {len(all_vertices)}, 内表面 {len(inner_idx)} / 外表面 {len(outer_idx)}")
                print(f"  内表面: 最大位移 {np.max(inner_disp):.2f}mm, 平均 {np.mean(inner_disp):.2f}mm")
                print(f"  外表面: 最大位移 {np.max(outer_disp):.2f}mm, 平均 {np.mean(outer_disp):.2f}mm")
            else:
                print(f"[方向拉伸] 总顶点 {len(all_vertices)}, 全部为内表面顶点")
                print(f"  最大位移 {np.max(inner_disp):.2f}mm")
            self._preview_vertices = deformed
            self._deformed_inner_vertices = deformed[inner_idx].copy()
            self._apply_vertex_update_full(deformed)
        else:
            deformed = self.deformation_engine.apply(
                self._deformed_inner_vertices, params
            )
            self._preview_vertices = deformed
            self._apply_vertex_update(deformed)
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
            if params.mode == "directional":
                all_vertices = self._get_all_brace_vertices()
                final = self.deformation_engine.apply(all_vertices, params)
            else:
                final = self.deformation_engine.apply(
                    self._deformed_inner_vertices, params
                )

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

        if params.mode == "directional":
            self._deformed_inner_vertices = final.copy()[self._inner_vertex_indices.astype(int)]
            self._apply_vertex_update_full(final)
        else:
            self._deformed_inner_vertices = final.copy()
            self._apply_vertex_update(final)

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
        if self._preview_vertices is None and self.deformation_state.step_count == 0:
            self.status_bar.showMessage("当前没有可还原的操作")
            return

        self._preview_vertices = None
        self._replay_deformations()
        self._render()
        self.status_bar.showMessage("已还原预览，回到变形前状态")

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

        self._deformed_inner_vertices = self._base_inner_vertices.copy()
        all_vertices = self._get_all_brace_vertices()

        for step in self.deformation_state.get_params_list():
            if step.mode == "directional":
                deformed_all = self.deformation_engine.apply(all_vertices, step)
                self._deformed_inner_vertices = deformed_all[self._inner_vertex_indices.astype(int)].copy()
                self._apply_vertex_update_full(deformed_all)
                all_vertices = deformed_all
            else:
                self._deformed_inner_vertices = self.deformation_engine.apply(
                    self._deformed_inner_vertices, step
                )
                self._apply_vertex_update(self._deformed_inner_vertices)

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
