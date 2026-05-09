"""模型对比与仿射变换映射 — UI 操作"""

import os
from typing import Optional

import numpy as np
import vtk
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from src.core.model_mapper import (
    compute_mapping, apply_transform_to_polydata,
    export_transformed_stl, print_mapping_report,
)


class _MappingWorker(QThread):
    """后台线程：计算模型映射"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, filepath_a: str, filepath_b: str):
        super().__init__()
        self.filepath_a = filepath_a
        self.filepath_b = filepath_b

    def run(self):
        try:
            self.progress.emit("正在加载模型...")
            self.progress.emit("正在计算 ICP 对齐...")
            result = compute_mapping(self.filepath_a, self.filepath_b)
            self.progress.emit("映射计算完成")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ActionsMappingMixin:
    """Mixin: 模型对比与仿射变换映射"""

    _map_file_a: Optional[str] = None
    _map_file_b: Optional[str] = None
    _map_result = None
    _map_model_type: str = "foot"  # "foot" 或 "brace"

    def _select_map_model_a(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择参考模型 A", "", "STL Files (*.stl)"
        )
        if filepath:
            self._map_file_a = filepath
            self.lbl_map_a.setText(f"A (参考): {os.path.basename(filepath)}")

    def _select_map_model_b(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择目标模型 B", "", "STL Files (*.stl)"
        )
        if filepath:
            self._map_file_b = filepath
            self.lbl_map_b.setText(f"B (目标): {os.path.basename(filepath)}")

    def _on_map_type_changed(self):
        idx = self.combo_map_type.currentIndex()
        self._map_model_type = "foot" if idx == 0 else "brace"

    def _run_mapping(self):
        if not self._map_file_a:
            self.status_bar.showMessage("请先选择参考模型 A")
            return
        if not self._map_file_b:
            self.status_bar.showMessage("请先选择目标模型 B")
            return

        self.lbl_map_matrix.setText("仿射矩阵: 计算中...")
        self.lbl_map_residual.setText("残差: 计算中...")
        self.map_result_text.setText("计算中...\n\n大模型可能需要几分钟。")

        self._map_worker = _MappingWorker(self._map_file_a, self._map_file_b)
        self._map_worker.progress.connect(self.status_bar.showMessage)
        self._map_worker.finished.connect(self._on_mapping_done)
        self._map_worker.error.connect(self._on_mapping_error)
        self._map_worker.start()

    def _on_mapping_done(self, result):
        self._map_result = result

        # 显示仿射矩阵
        M = result.affine_matrix
        matrix_lines = []
        for i in range(4):
            vals = "  ".join(f"{M[i,j]:>10.6f}" for j in range(4))
            matrix_lines.append(f"  [{vals}]")
        self.lbl_map_matrix.setText("仿射矩阵:\n" + "\n".join(matrix_lines))

        # 显示残差统计
        p = result.percentile_residuals
        self.lbl_map_residual.setText(
            f"平均残差: {result.mean_residual:.4f} mm\n"
            f"RMS 残差: {result.rms_residual:.4f} mm\n"
            f"最大残差: {result.max_residual:.4f} mm\n"
            f"P50: {p[2]:.4f} mm  P95: {p[5]:.4f} mm"
        )

        # 显示详细报告
        label_a = os.path.basename(self._map_file_a)
        label_b = os.path.basename(self._map_file_b)
        from io import StringIO
        import sys
        buf = StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        print_mapping_report(result, label_a, label_b)
        sys.stdout = old_stdout
        self.map_result_text.setText(buf.getvalue())

        self.status_bar.showMessage(
            f"映射完成: {label_a} → {label_b}, "
            f"平均残差={result.mean_residual:.4f}mm"
        )

    def _on_mapping_error(self, error_msg: str):
        self.map_result_text.setText(f"映射失败:\n{error_msg}")
        self.status_bar.showMessage("模型映射失败")

    def _preview_transformed(self):
        """在 3D 视图中预览变换后的模型 A"""
        if self._map_result is None:
            self.status_bar.showMessage("请先计算映射")
            return

        if self.brace_model is None and self.foot_model is None:
            self.status_bar.showMessage("请先加载至少一个模型")
            return

        # 对参考模型应用完整变换（ICP + 仿射）
        M = self._map_result.alignment.matrix_4x4 @ self._map_result.affine_matrix
        source_poly = vtk.vtkPolyData()

        # 使用模型 A 的原始文件
        from src.core.model_loader import load_stl
        model_a = load_stl(self._map_file_a)

        transformed = apply_transform_to_polydata(model_a.polydata, M)

        # 在 3D 视图中显示变换后的模型
        # 清除旧的预览
        if hasattr(self, '_map_preview_actor') and self._map_preview_actor:
            try:
                self.scene.renderer.RemoveActor(self._map_preview_actor)
            except Exception:
                pass

        # 创建新 actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(transformed)
        mapper.SetScalarRange(0, 1)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 红色预览
        actor.GetProperty().SetOpacity(0.5)
        self.scene.renderer.AddActor(actor)
        self._map_preview_actor = actor

        self._render()
        self.status_bar.showMessage("已显示变换预览（红色半透明）")

    def _export_transformed_model(self):
        """导出变换后的模型"""
        if self._map_result is None:
            self.status_bar.showMessage("请先计算映射")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出变换后的 STL", "", "STL Files (*.stl)"
        )
        if not filepath:
            return

        try:
            self.status_bar.showMessage("正在导出...")
            M = (self._map_result.alignment.matrix_4x4
                 @ self._map_result.affine_matrix)

            from src.core.model_loader import load_stl
            model_a = load_stl(self._map_file_a)
            transformed = apply_transform_to_polydata(model_a.polydata, M)

            writer = vtk.vtkSTLWriter()
            writer.SetFileName(filepath)
            writer.SetFileTypeToBinary()
            writer.SetInputData(transformed)
            writer.Write()

            self.status_bar.showMessage(
                f"已导出变换模型: {os.path.basename(filepath)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"无法导出:\n{e}")
