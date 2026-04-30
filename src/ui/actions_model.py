"""模型加载、STEP 转换相关操作"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import vtk
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QProcess
from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QProgressDialog,
)

from src.core.model_loader import ModelData, load_stl
from src.core.transform_manager import TransformManager
from src.core.step_converter import step_to_stl, stl_to_step


class FreeCADWorker(QThread):
    """后台线程：执行 FreeCAD STEP→STL 转换（使用 QProcess 支持取消）"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    FREECAD_CMD = "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd"

    def __init__(self, step_path: str, stl_path: str, parent=None):
        super().__init__(parent)
        self.step_path = step_path
        self.stl_path = stl_path
        self._process = None
        self._temp_script = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        if self._process and self._process.state() == QProcess.Running:
            self._process.kill()

    def run(self):
        import tempfile

        if self._cancelled:
            return

        if os.path.exists(self.stl_path):
            step_mtime = os.path.getmtime(self.step_path)
            stl_mtime = os.path.getmtime(self.stl_path)
            if stl_mtime >= step_mtime:
                self.progress.emit("使用已转换的 STL 缓存")
                self.finished.emit(self.stl_path)
                return

        script = f"""
import Part, os, FreeCAD, sys
step_path = {self.step_path!r}
stl_path = {self.stl_path!r}
try:
    shape = Part.read(step_path)
    doc = FreeCAD.newDocument('StepConvert')
    obj = doc.addObject('Part::Feature', 'Shape')
    obj.Shape = shape
    doc.recompute()
    shape.tessellate(0.05)
    Part.export([obj], stl_path)
except Exception as e:
    print(f'ERROR: {{e}}', file=sys.stderr)
    sys.exit(1)
finally:
    try:
        FreeCAD.closeDocument(doc.Name)
    except Exception:
        pass
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script)
            f.flush()
            self._temp_script = f.name

        try:
            self._process = QProcess()
            self._process.setProcessChannelMode(QProcess.MergedChannels)

            output_lines = []

            def on_ready_read():
                data = self._process.readAllStandardOutput().data().decode('utf-8', errors='replace')
                output_lines.append(data)
                if "Reading" in data or "read" in data.lower():
                    self.progress.emit("正在读取 STEP 几何...")
                elif "Meshing" in data or "tessellate" in data:
                    self.progress.emit("正在网格化导出...")

            self._process.readyReadStandardOutput.connect(on_ready_read)

            self._process.start(
                self.FREECAD_CMD, [self._temp_script]
            )
            self._process.waitForFinished(-1)

            if self._cancelled:
                self.error.emit("用户已取消转换")
                return

            exit_code = self._process.exitCode()
            if exit_code != 0:
                stderr = self._process.readAllStandardError().data().decode(
                    'utf-8', errors='replace'
                )
                raise RuntimeError(
                    f"FreeCAD 执行失败 (code={exit_code}):\n{stderr[-2000:]}"
                )
            self.finished.emit(self.stl_path)
        except RuntimeError:
            raise
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if self._temp_script and os.path.exists(self._temp_script):
                os.unlink(self._temp_script)


class _BraceToStepWorker(QThread):
    """后台线程：将当前 VTK 护具 polydata 转换为 STEP 再网格化"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, polydata: vtk.vtkPolyData, stl_path: str):
        super().__init__()
        self.polydata = polydata
        self.stl_path = stl_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            import tempfile

            temp_stl = self.stl_path + ".orig.stl"
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(temp_stl)
            writer.SetInputData(self.polydata)
            writer.Write()

            if self._cancelled:
                return

            self.progress.emit("正在转换为 STEP...")
            step_path = stl_to_step(temp_stl, tolerance=0.1)

            if self._cancelled:
                return

            self.progress.emit("正在高精度网格化...")
            step_to_stl(step_path, self.stl_path, linear_deflection=0.05)

            if self._cancelled:
                return

            for f in [temp_stl, step_path]:
                if os.path.exists(f):
                    os.unlink(f)

            self.finished.emit(self.stl_path)
        except Exception as e:
            if self._cancelled:
                self.error.emit("用户已取消转换")
            else:
                self.error.emit(str(e))


class ActionsModelMixin:
    """Mixin: 模型加载、STEP 转换"""

    def _load_foot(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择足部 STL 文件", "", "STL Files (*.stl)"
        )
        if not filepath:
            return
        try:
            self.foot_model = load_stl(filepath)
            self.lbl_foot.setText(f"足部模型: {self.foot_model.name}")

            self.scene.clear_min_max_indicators()

            if self.scene._foot_actor:
                self.scene.renderer.RemoveActor(self.scene._foot_actor)
            self.scene.add_foot_model(self.foot_model.polydata)

            self.distance_calc = DistanceCalculator(
                self.foot_model.polydata
            )

            if self.foot_model and self.brace_model:
                self._update_axes()
            else:
                self.scene.fit_camera_to_models()
            self.status_bar.showMessage(
                f"已加载足部模型: {self.foot_model.name} "
                f"({self.foot_model.vertex_count:,} 顶点)"
            )
            self._update_model_info()
            self._render()
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法加载足部模型:\n{e}")

    def _load_brace(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择护具 STL 文件", "", "STL Files (*.stl)"
        )
        if not filepath:
            return

        self._brace_step_filepath = None
        self._prepare_and_load(filepath)

    def _load_brace_step(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择护具 STEP 文件", "", "STEP Files (*.step *.stp)"
        )
        if not filepath:
            return
        self._load_brace_step_from_file(filepath)

    def _prepare_and_load(self, stl_path: str):
        try:
            self._refresh_inner_highlight()
            self._inner_cells = None
            self._inner_regions = []
            self._inner_vertex_indices = None
            self._region_actor_map.clear()
            self._clear_region_lists()

            self.scene.clear_min_max_indicators()
            self.deformation_state.clear()
            self.deformation_engine = None
            self._deformed_inner_vertices = None
            self._original_inner_vertices = None
            self._base_inner_vertices = None

            self.brace_model = load_stl(stl_path)
            self.lbl_brace.setText(f"护具模型: {self.brace_model.name}")
            self.brace_transform = TransformManager(
                center=self.brace_model.centroid
            )

            # 保存原始护具副本（用于还原操作）
            self._original_brace_polydata = vtk.vtkPolyData()
            self._original_brace_polydata.DeepCopy(self.brace_model.polydata)

            if self.scene._brace_actor:
                self.scene.renderer.RemoveActor(self.scene._brace_actor)
            self.scene.add_brace_model(self.brace_model.polydata)

            if self.foot_model and self.brace_model:
                self._update_axes()
            else:
                self.scene.fit_camera_to_models()
            self.status_bar.showMessage(
                f"已加载护具模型: {self.brace_model.name} "
                f"({self.brace_model.vertex_count:,} 顶点)"
            )
            self._update_model_info()
            self._render()

            self._current_brace_filepath = stl_path

            self._auto_load_brace_positions(stl_path)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法加载护具模型:\n{e}")

    def _convert_brace_to_step(self):
        if self.brace_model is None:
            QMessageBox.warning(self, "警告", "请先在「模型」标签页加载护具 STL")
            return

        brace_path = self._current_brace_filepath
        if brace_path and os.path.exists(brace_path):
            base = os.path.splitext(brace_path)[0]
            candidates = [base + ".step", base + ".stp"]
            existing_step = None
            for c in candidates:
                if os.path.exists(c):
                    existing_step = c
                    break

            if existing_step:
                reply = QMessageBox.question(
                    self, "STEP 文件已存在",
                    f"已找到对应的 STEP 文件：\n{Path(existing_step).name}\n\n"
                    f"是否直接加载 STEP 而非重新转换？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    self._load_brace_step_from_file(existing_step)
                    return

        try:
            self._step_progress = QProgressDialog(
                "正在转换为 STEP 并网格化...",
                "取消", 0, 0, self
            )
            self._step_progress.setWindowTitle("STEP 转换")
            self._step_progress.setWindowModality(Qt.WindowModal)
            self._step_progress.setMinimumDuration(0)
            self._step_progress.show()

            stl_temp = "/tmp/anklepro_brace_converted.stl"

            self._step_worker = _BraceToStepWorker(
                self.brace_model.polydata, stl_temp
            )
            self._step_progress.canceled.connect(self._step_worker.cancel)
            self._step_worker.progress.connect(
                self._step_progress.setLabelText
            )
            self._step_worker.finished.connect(
                self._on_convert_brace_to_step_done
            )
            self._step_worker.error.connect(self._on_step_conversion_error)
            self._step_worker.start()
        except Exception as e:
            QMessageBox.critical(self, "转换失败", f"无法转换为 STEP:\n{e}")

    def _on_convert_brace_to_step_done(self, stl_path: str):
        self._step_progress.close()
        self._brace_step_filepath = stl_path
        self._prepare_and_load(stl_path)
        self.status_bar.showMessage(
            f"已加载护具 (STEP 高精度): {Path(stl_path).name}"
        )

    def _load_brace_step_from_file(self, step_path: str):
        self._brace_step_filepath = step_path
        stl_path = step_path.replace(".step", "_mesh.stl").replace(
            ".stp", "_mesh.stl"
        )
        if stl_path == step_path:
            stl_path = step_path + "_mesh.stl"

        self._step_progress = QProgressDialog(
            "正在从 STEP 加载护具...",
            "取消", 0, 0, self
        )
        self._step_progress.setWindowTitle("STEP 转换")
        self._step_progress.setWindowModality(Qt.WindowModal)
        self._step_progress.setMinimumDuration(0)
        self._step_progress.show()

        self._step_worker = FreeCADWorker(step_path, stl_path)
        self._step_progress.canceled.connect(self._step_worker.cancel)
        self._step_worker.progress.connect(
            self._step_progress.setLabelText
        )
        self._step_worker.finished.connect(self._on_step_conversion_done)
        self._step_worker.error.connect(self._on_step_conversion_error)
        self._step_worker.start()

    def _on_step_conversion_done(self, stl_path: str):
        self._step_progress.close()

        self._refresh_inner_highlight()
        self._inner_cells = None
        self._inner_regions = []
        self._inner_vertex_indices = None
        self._region_actor_map.clear()
        self.region_list.clear()
        self.scene.clear_min_max_indicators()
        self.deformation_state.clear()
        self.deformation_engine = None
        self._deformed_inner_vertices = None
        self._original_inner_vertices = None
        self._base_inner_vertices = None

        self._prepare_and_load(stl_path)

        self.status_bar.showMessage(
            f"已加载护具 STEP: {Path(self._brace_step_filepath).name} "
            f"(→ {Path(stl_path).name})"
        )

    def _on_step_conversion_error(self, error_msg: str):
        self._step_progress.close()
        QMessageBox.critical(self, "加载失败", f"无法加载 STEP 文件:\n{error_msg}")
        self.status_bar.showMessage("STEP 加载失败")


# Deferred import to avoid circular dependency
from src.core.distance_calculator import DistanceCalculator
