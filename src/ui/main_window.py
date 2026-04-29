"""主窗口 — 包含 3D 视图、控制面板和状态栏"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import vtk
from PyQt5.QtCore import Qt, QTimer, QEvent, QThread, pyqtSignal, QProcess
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from src.config import (
    DEFAULT_MOVE_STEP,
    FINE_MOVE_STEP,
    DEFAULT_ROTATE_STEP,
    FINE_ROTATE_STEP,
    PRESET_THRESHOLDS,
    DEFAULT_PRESET,
)
from src.version import __version__
from src.core.model_loader import ModelData, load_stl
from src.core.transform_manager import TransformManager
from src.core.distance_calculator import DistanceCalculator
from src.core.optimizer import BraceOptimizer
from src.core.radial_distance_calculator import RadialDistanceCalculator, AxisCalculator
from src.core.surface_picker import (
    identify_inner_surface,
    extract_inner_vertices,
    get_transformed_vertices,
    find_connected_regions,
)
from src.core.position_file_manager import PositionFileManager
from src.core.distance_measurement import DistanceMeasurement
from src.core.deformation_engine import DeformationEngine, DeformationParams
from src.core.deformation_state import DeformationState, DeformationStep
from src.core.step_converter import step_to_stl, stl_to_step, get_step_info
from src.render.scene_manager import SceneManager
from src.render.interactor_style import BraceCameraInteractorStyle
from src.ui.ui_builder import UIBuilderMixin
from src.ui.actions import ActionsMixin


class FreeCADWorker(QThread):
    """后台线程：执行 FreeCAD STEP→STL 转换（使用 QProcess 支持取消）"""
    progress = pyqtSignal(str)  # 进度消息
    finished = pyqtSignal(str)  # 输出 STL 路径
    error = pyqtSignal(str)     # 错误信息

    FREECAD_CMD = "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd"

    def __init__(self, step_path: str, stl_path: str, parent=None):
        super().__init__(parent)
        self.step_path = step_path
        self.stl_path = stl_path
        self._process = None
        self._temp_script = None
        self._cancelled = False

    def cancel(self):
        """取消转换，终止 FreeCAD 进程"""
        self._cancelled = True
        if self._process and self._process.state() == QProcess.Running:
            self._process.kill()

    def run(self):
        if self._cancelled:
            return

        # 检查缓存
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
    finished = pyqtSignal(str)  # 输出 STL 路径
    error = pyqtSignal(str)

    def __init__(self, polydata: vtk.vtkPolyData, stl_path: str):
        super().__init__()
        self.polydata = polydata
        self.stl_path = stl_path
        self._cancelled = False

    def cancel(self):
        """取消转换"""
        self._cancelled = True

    def run(self):
        try:
            import tempfile

            # 1. 导出临时 STL
            temp_stl = self.stl_path + ".orig.stl"
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(temp_stl)
            writer.SetInputData(self.polydata)
            writer.Write()

            if self._cancelled:
                return

            self.progress.emit("正在转换为 STEP...")
            # 2. STL → STEP
            step_path = stl_to_step(temp_stl, tolerance=0.1)

            if self._cancelled:
                return

            self.progress.emit("正在高精度网格化...")
            # 3. STEP → 高精度 STL
            step_to_stl(step_path, self.stl_path, linear_deflection=0.05)

            if self._cancelled:
                return

            # 清理临时文件
            for f in [temp_stl, step_path]:
                if os.path.exists(f):
                    os.unlink(f)

            self.finished.emit(self.stl_path)
        except Exception as e:
            if self._cancelled:
                self.error.emit("用户已取消转换")
            else:
                self.error.emit(str(e))



class MainWindow(UIBuilderMixin, ActionsMixin, QMainWindow):
    """AnklePro 主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"AnklePro — 足部护具贴合度分析 {__version__}")
        # 根据屏幕尺寸调整窗口大小，确保不超出屏幕
        screen = self.screen().geometry()
        window_width = min(1400, int(screen.width() * 0.9))
        window_height = min(int(screen.height() * 0.85), 800)
        self.resize(window_width, window_height)

        # ---- 状态 ----
        self.foot_model: Optional[ModelData] = None
        self.brace_model: Optional[ModelData] = None
        self.brace_transform: TransformManager = TransformManager()
        self.distance_calc: Optional[DistanceCalculator] = None
        self.radial_calc: Optional[RadialDistanceCalculator] = None
        self.current_distances: Optional[np.ndarray] = None
        self.current_thresholds = PRESET_THRESHOLDS[DEFAULT_PRESET]
        self.is_fine_mode = False
        self.move_step = DEFAULT_MOVE_STEP
        self.rotate_step = DEFAULT_ROTATE_STEP
        self._wireframe = False
        self._last_calc_position = np.zeros(3)
        self._pending_recalc = False

        # 测量工具状态
        self._measure_tool_active = False
        self._measure_calc = DistanceMeasurement()

        # 位置保存（5 个槽位）
        self.saved_positions: list = [None] * 5  # 每个元素为 (translation, rotation) 元组

        # 内侧面选取
        self._inner_cells: Optional[np.ndarray] = None  # 内侧面 cell 索引
        self._inner_vertex_indices: Optional[np.ndarray] = None  # 内侧面顶点索引
        self._inner_regions: list = []  # 连通区域列表
        self._region_actors: list = []  # 每个区域的 Actor（已废弃，保留用于兼容）
        self._region_actor_map: dict = {}  # region_idx -> actor 映射
        self._region_labels: list = []  # 每个区域的标签 Actor

        # 位置文件管理器
        self.position_file_manager = PositionFileManager()

        # 当前护具文件路径（用于关联位置文件）
        self._current_brace_filepath: Optional[str] = None

        # 尺寸修改
        self.deformation_state = DeformationState()
        self.deformation_engine: Optional[DeformationEngine] = None
        self._deform_mode = "normal"  # "normal", "directional", "radial", "adaptive"

        # 变形选点状态
        self._deform_point_idx: int = -1  # 当前选点在内表面数组中的索引（-1=未设置）
        self._inner_adjacency: dict = {}  # 内表面网格邻接表 {idx: [neighbor_indices]}
        self._base_inner_vertices: Optional[np.ndarray] = None  # 变形引擎初始化时的内表面顶点位置（不变）
        self._original_inner_vertices: Optional[np.ndarray] = None  # 当前内表面顶点位置（随变形累积）
        self._deformed_inner_vertices: Optional[np.ndarray] = None  # 变形后的内表面顶点位置
        self._preview_vertices: Optional[np.ndarray] = None  # 当前预览的顶点位置
        self._brace_step_filepath: Optional[str] = None  # 护具 STEP 文件路径

        self._setup_ui()
        self._setup_menu()
        self._setup_vtk()
        self.scene.add_help_text(True)

        self._setup_ui()
        self._setup_menu()
        self._setup_vtk()
        self.scene.add_help_text(True)

    def _setup_menu(self):
        """设置菜单栏"""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Foot Model...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._load_foot)
        file_menu.addAction(open_action)

        open_brace_action = QAction("Open Brace Model...", self)
        open_brace_action.setShortcut("Ctrl+Shift+O")
        open_brace_action.triggered.connect(self._load_brace)
        file_menu.addAction(open_brace_action)

        file_menu.addSeparator()

        screenshot_action = QAction("Save Screenshot...", self)
        screenshot_action.setShortcut("Ctrl+S")
        screenshot_action.triggered.connect(self._screenshot)
        file_menu.addAction(screenshot_action)

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("View")
        reset_cam_action = QAction("Reset Camera", self)
        reset_cam_action.setShortcut("Ctrl+R")
        reset_cam_action.triggered.connect(self._reset_camera)
        view_menu.addAction(reset_cam_action)

        tools_menu = menubar.addMenu("Tools")
        calc_action = QAction("Compute Distance", self)
        calc_action.setShortcut("Ctrl+D")
        calc_action.triggered.connect(self._compute_distance)
        tools_menu.addAction(calc_action)

        measure_action = QAction("测量距离", self)
        measure_action.setShortcut("Ctrl+M")
        measure_action.setCheckable(True)
        measure_action.triggered.connect(self._toggle_measurement)
        tools_menu.addAction(measure_action)
        self.measure_action = measure_action

    def _setup_vtk(self):
        """初始化 VTK 渲染 + 自定义交互器"""
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.scene = SceneManager(self.render_window)

        # 使用自定义交互器样式（仅处理鼠标）
        self.iren = self.render_window.GetInteractor()
        self.custom_style = BraceCameraInteractorStyle(
            renderer=self.scene.renderer,
            main_window=self
        )
        self.custom_style.SetDefaultRenderer(self.scene.renderer)
        self.iren.SetInteractorStyle(self.custom_style)

        # 在 Qt 层面拦截键盘事件（确保不被 VTK 拦截）
        self.vtk_widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        """
        Qt 事件过滤器 — 拦截 VTK 控件的键盘和鼠标事件

        方向键 → 相机缩放/旋转视角
        字母键 → 护具平移/旋转
        鼠标点击/移动 → 测量工具
        """
        if obj is self.vtk_widget:
            # ---- 测量模式：鼠标事件 ----
            if self._measure_tool_active:
                # 动态计算 Qt→VTK 坐标缩放比（适配任意 DPI/Retina 显示器）
                qtw = self.vtk_widget.width()
                qth = self.vtk_widget.height()
                rw, rh = self.render_window.GetSize()
                sx = rw / max(qtw, 1)
                sy = rh / max(qth, 1)

                if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
                    x = int(event.pos().x() * sx)
                    y = rh - int(event.pos().y() * sy)  # VTK Y 轴原点在底部
                    self._on_measurement_click(x, y)
                    return True  # 消费事件，阻止 VTK 旋转相机
                if event.type() == event.MouseMove:
                    x = int(event.pos().x() * sx)
                    y = rh - int(event.pos().y() * sy)
                    self._on_measurement_move(x, y)
                    return False  # 不消费，允许 VTK 正常旋转

            # ---- 键盘事件 ----
            if event.type() == event.KeyPress:
                key = event.key()
                modifiers = event.modifiers()

                # ---- 方向键：变形选点导航 或 相机操作 ----
                if (self._deform_mode == "directional"
                        and self._deform_point_idx >= 0):
                    # 方向拉伸模式下，方向键控制选点移动
                    if key == Qt.Key_Up:
                        self._move_deform_point("up")
                        return True
                    if key == Qt.Key_Down:
                        self._move_deform_point("down")
                        return True
                    if key == Qt.Key_Left:
                        self._move_deform_point("left")
                        return True
                    if key == Qt.Key_Right:
                        self._move_deform_point("right")
                        return True

                # 默认：方向键控制相机
                if key == Qt.Key_Up:
                    camera = self.scene.renderer.GetActiveCamera()
                    camera.Zoom(1.1)
                    self._render()
                    return True
                if key == Qt.Key_Down:
                    camera = self.scene.renderer.GetActiveCamera()
                    camera.Zoom(0.9)
                    self._render()
                    return True
                if key == Qt.Key_Left:
                    camera = self.scene.renderer.GetActiveCamera()
                    camera.Azimuth(-5)
                    self._render()
                    return True
                if key == Qt.Key_Right:
                    camera = self.scene.renderer.GetActiveCamera()
                    camera.Azimuth(5)
                    self._render()
                    return True

                # ---- 字母键：护具操作 ----
                if self.scene._brace_actor is not None:
                    step = self.move_step
                    rot = self.rotate_step

                    if key == Qt.Key_F:
                        # 绕 Z 旋转 +step
                        self.brace_transform.rotate_z(rot)
                        self._update_brace_view()
                        return True
                    if key == Qt.Key_G:
                        # 绕 Z 旋转 -step
                        self.brace_transform.rotate_z(-rot)
                        self._update_brace_view()
                        return True
                    if key == Qt.Key_H:
                        # 绕 Y 旋转 -step
                        self.brace_transform.rotate_y(-rot)
                        self._update_brace_view()
                        return True
                    if key == Qt.Key_J:
                        # 绕 Y 旋转 +step
                        self.brace_transform.rotate_y(rot)
                        self._update_brace_view()
                        return True
                    if key == Qt.Key_N:
                        # 绕 X 旋转 -step
                        self.brace_transform.rotate_x(-rot)
                        self._update_brace_view()
                        return True
                    if key == Qt.Key_M:
                        # 绕 X 旋转 +step
                        self.brace_transform.rotate_x(rot)
                        self._update_brace_view()
                        return True
                    if key == Qt.Key_B:
                        # 平移 X +step
                        self.brace_transform.translate(step, 0, 0)
                        self._update_brace_view()
                        return True
                    if key == Qt.Key_V:
                        # 平移 X -step
                        self.brace_transform.translate(-step, 0, 0)
                        self._update_brace_view()
                        return True
                    if key == Qt.Key_U:
                        # 平移 Y +step
                        self.brace_transform.translate(0, step, 0)
                        self._update_brace_view()
                        return True
                    if key == Qt.Key_I:
                        # 平移 Y -step
                        self.brace_transform.translate(0, -step, 0)
                        self._update_brace_view()
                        return True
                    if key == Qt.Key_R:
                        # 平移 Z +step
                        self.brace_transform.translate(0, 0, step)
                        self._update_brace_view()
                        return True
                    if key == Qt.Key_T:
                        # 平移 Z -step
                        self.brace_transform.translate(0, 0, -step)
                        self._update_brace_view()
                        return True

                # D 键 — 线框/实体切换
                if key == Qt.Key_D:
                    self._toggle_wireframe()
                    return True

                # Ctrl+Z / Cmd+Z — 撤销
                if key == Qt.Key_Z and (
                    (modifiers & Qt.ControlModifier)
                    or (modifiers & Qt.MetaModifier)
                ):
                    self._undo_transform()
                    return True

                # Escape — 退出测量模式
                if key == Qt.Key_Escape and self._measure_tool_active:
                    self._toggle_measurement(False)
                    return True

        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        self.vtk_widget.close()
        super().closeEvent(event)


    def _render(self):
        """刷新渲染"""
        if self.render_window:
            self.render_window.Render()


def _polydata_to_vertices(polydata: vtk.vtkPolyData) -> np.ndarray:
    """提取 vtkPolyData 的顶点数组 (N, 3)"""
    points = polydata.GetPoints()
    n = points.GetNumberOfPoints()
    vertices = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        vertices[i] = points.GetPoint(i)
    return vertices

