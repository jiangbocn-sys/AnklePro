"""位置保存、加载、清空、自动加载相关操作"""

import numpy as np
from PyQt5.QtWidgets import (
    QListWidgetItem,
    QMessageBox,
)


class ActionsPositionMixin:
    """Mixin: 位置保存/加载/管理"""

    def _auto_load_brace_positions(self, filepath: str):
        result = self.position_file_manager.load_positions(filepath)

        if result["found"]:
            self.saved_positions = result["positions"]
            self._refresh_position_list()

            if result.get("inner_surface_indices") is not None:
                self._inner_vertex_indices = result["inner_surface_indices"]
            if result.get("inner_regions") is not None:
                self._inner_regions = result["inner_regions"]

            count = sum(1 for p in self.saved_positions if p is not None)
            self.status_bar.showMessage(f"已自动加载 {count} 个保存的位置")
        else:
            self.saved_positions = [None] * 5
            self._refresh_position_list()

    def _refresh_position_list(self):
        self.position_list.clear()
        for i, pos in enumerate(self.saved_positions):
            if pos is not None:
                translation, rotation = pos
                item_text = (
                    f"位置 {i+1}: "
                    f"X={translation[0]:+.1f} Y={translation[1]:+.1f} Z={translation[2]:+.1f} | "
                    f"Rx={rotation[0]:+.1f}° Ry={rotation[1]:+.1f}° Rz={rotation[2]:+.1f}°"
                )
                item = QListWidgetItem(item_text)
            else:
                item = QListWidgetItem(f"位置 {i+1}: (空)")
            self.position_list.addItem(item)

    def _save_position(self):
        if self.brace_model is None:
            QMessageBox.warning(self, "警告", "请先加载护具模型")
            return

        selected_items = self.position_list.selectedItems()
        if selected_items:
            slot_idx = self.position_list.row(selected_items[0])
        else:
            slot_idx = 0
            for i, pos in enumerate(self.saved_positions):
                if pos is None:
                    slot_idx = i
                    break

        translation = self.brace_transform.get_translation()
        rotation = self.brace_transform.get_rotation()
        self.saved_positions[slot_idx] = (translation.copy(), rotation.copy())
        self._refresh_position_list()
        self.position_list.setCurrentRow(slot_idx)

        if self._current_brace_filepath:
            self.position_file_manager.save_positions(
                self._current_brace_filepath,
                self.saved_positions,
                self._inner_vertex_indices,
                self._inner_regions,
            )

        self.status_bar.showMessage(f"位置已保存到槽位 {slot_idx + 1} 并写入文件")

    def _load_position(self):
        selected_items = self.position_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要加载的位置")
            return

        slot_idx = self.position_list.row(selected_items[0])
        if self.saved_positions[slot_idx] is None:
            QMessageBox.information(self, "提示", "该槽位为空")
            return

        translation, rotation = self.saved_positions[slot_idx]

        self.brace_transform.reset()
        self.brace_transform.translate(*translation.tolist())
        if np.any(rotation):
            self.brace_transform.rotate_x(rotation[0])
            self.brace_transform.rotate_y(rotation[1])
            self.brace_transform.rotate_z(rotation[2])

        self._update_brace_view()
        self._render()
        self.status_bar.showMessage(f"已加载位置 {slot_idx + 1}")

        if self.current_distances is not None:
            reply = QMessageBox.question(
                self, "重新计算距离",
                "是否根据新位置重新计算距离？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._compute_distance()

    def _clear_position(self):
        selected_items = self.position_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要清空的位置")
            return

        slot_idx = self.position_list.row(selected_items[0])
        self.saved_positions[slot_idx] = None
        self._refresh_position_list()
        self.status_bar.showMessage(f"位置 {slot_idx + 1} 已清空")

    def _on_position_double_clicked(self, item):
        slot_idx = self.position_list.row(item)
        if self.saved_positions[slot_idx] is None:
            self.status_bar.showMessage(f"位置 {slot_idx + 1} 为空")
            return

        translation, rotation = self.saved_positions[slot_idx]

        self.brace_transform.reset()
        self.brace_transform.translate(*translation.tolist())
        if np.any(rotation):
            self.brace_transform.rotate_x(rotation[0])
            self.brace_transform.rotate_y(rotation[1])
            self.brace_transform.rotate_z(rotation[2])

        self._update_brace_view()
        self._render()

        if self.current_distances is not None:
            reply = QMessageBox.question(
                self, "重新计算距离",
                "是否根据新位置重新计算距离？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._compute_distance()
