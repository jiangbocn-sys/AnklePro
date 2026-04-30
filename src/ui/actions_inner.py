"""内侧面选取、区域管理相关操作"""

from typing import Optional

import numpy as np
import vtk
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidgetItem

from src.core.surface_picker import (
    identify_inner_surface,
    extract_inner_vertices,
    find_connected_regions,
)


class ActionsInnerMixin:
    """Mixin: 内侧面选取、区域列表管理"""

    def _pick_inner_surface(self):
        if self.brace_model is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请先加载护具模型")
            return

        self.status_bar.showMessage("正在识别内侧面...")

        self._inner_cells = identify_inner_surface(
            self.brace_model.polydata
        )
        self._inner_vertex_indices = extract_inner_vertices(
            self.brace_model.polydata, self._inner_cells
        )

        self._build_inner_adjacency()

        self._inner_regions = find_connected_regions(
            self._inner_cells,
            self.brace_model.polydata,
            angle_threshold=75.0,
            min_region_size=150,
        )

        self._refresh_inner_highlight()
        self._render()

        self._clear_region_lists()
        colors = [
            (255, 50, 50),    (50, 200, 50),    (50, 100, 255),
            (255, 140, 50),   (200, 50, 255),   (50, 255, 255),
            (255, 100, 180),  (150, 255, 100),  (100, 180, 255),
            (255, 200, 100),
        ]
        for i, region in enumerate(self._inner_regions):
            color = colors[i % len(colors)]
            item = QListWidgetItem(f"区域 {i+1} ({len(region):,} 面)")
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, (i, color))
            self.region_list.addItem(item)

        n_inner = len(self._inner_cells)
        n_vertices = len(self._inner_vertex_indices)
        n_total = self.brace_model.triangle_count
        n_regions = len(self._inner_regions)
        self.lbl_inner_stats.setText(
            f"内侧面: {n_inner:,} 三角面 / {n_total:,} 总面\n"
            f"顶点数: {n_vertices:,} | 区域数: {n_regions}"
        )
        self.status_bar.showMessage(
            f"已选取内侧面: {n_inner:,} 三角面 / {n_total:,} 总面 "
            f"({n_inner/n_total*100:.1f}%), {n_vertices:,} 顶点, "
            f"{n_regions} 个区域 (已过滤小于 150 面的碎片)"
        )

    def _on_region_toggled(self, item: QListWidgetItem):
        region_idx, color = item.data(Qt.UserRole)
        region_cells = self._inner_regions[region_idx]

        highlight_color = (100, 180, 255)

        if item.checkState() == Qt.Checked:
            if region_idx not in self._region_actor_map:
                actor = self.scene._create_inner_surface_actor(
                    self.brace_model.polydata, region_cells, highlight_color
                )
                self.scene.renderer.AddActor(actor)
                self._region_actor_map[region_idx] = actor
        else:
            if region_idx in self._region_actor_map:
                actor = self._region_actor_map[region_idx]
                self.scene.renderer.RemoveActor(actor)
                del self._region_actor_map[region_idx]

        self._render()

    def _select_all_regions(self):
        for i in range(self.region_list.count()):
            self.region_list.item(i).setCheckState(Qt.Checked)

    def _deselect_all_regions(self):
        for i in range(self.region_list.count()):
            self.region_list.item(i).setCheckState(Qt.Unchecked)

    def _refresh_inner_highlight(self):
        for actor in self._region_actor_map.values():
            self._safe_remove_actor(actor)
        self._region_actor_map.clear()
        for label in self._region_labels:
            self._safe_remove_actor(label)
        self._region_labels = []
        self.scene.clear_inner_surface_actors()

    def _get_selected_inner_vertices(self) -> Optional[np.ndarray]:
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

        selected_indices = np.array(sorted(selected_vertex_set), dtype=np.int64)
        self._current_selected_indices = selected_indices

        from src.core.surface_picker import get_transformed_vertices
        return get_transformed_vertices(
            self.brace_model.polydata,
            selected_indices,
            self.brace_transform.get_matrix(),
        )

    def _build_inner_adjacency(self):
        if self._inner_vertex_indices is None:
            return

        global_to_inner = {int(idx): i for i, idx in enumerate(self._inner_vertex_indices)}
        adjacency = {i: [] for i in range(len(self._inner_vertex_indices))}

        cell_array = self.brace_model.polydata.GetPolys()
        cell_array.InitTraversal()
        id_list = vtk.vtkIdList()
        while cell_array.GetNextCell(id_list):
            cell_global_ids = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]
            cell_inner_ids = []
            for gid in cell_global_ids:
                if gid in global_to_inner:
                    cell_inner_ids.append(global_to_inner[gid])
            for i, aid in enumerate(cell_inner_ids):
                for j, other in enumerate(cell_inner_ids):
                    if i != j and other not in adjacency[aid]:
                        adjacency[aid].append(other)

        self._inner_adjacency = adjacency
