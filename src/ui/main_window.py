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
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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


class MainWindow(QMainWindow):
    """AnklePro 主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnklePro — 足部护具贴合度分析 v0.1.0")
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
        self._deform_direction = "shrink"  # "shrink" or "expand"
        self._original_inner_vertices: Optional[np.ndarray] = None  # 原始内表面顶点位置
        self._deformed_inner_vertices: Optional[np.ndarray] = None  # 变形后的内表面顶点位置
        self._preview_vertices: Optional[np.ndarray] = None  # 当前预览的顶点位置
        self._brace_step_filepath: Optional[str] = None  # 护具 STEP 文件路径

        self._setup_ui()
        self._setup_menu()
        self._setup_vtk()
        self.scene.add_help_text(True)

    # ================================================================
    # UI 设置
    # ================================================================

    def _setup_ui(self):
        """设置主窗口布局"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # --- 左侧: 3D视图 ---
        self.vtk_widget = QVTKRenderWindowInteractor(central)
        main_layout.addWidget(self.vtk_widget, stretch=4)

        # --- 右侧: 控制面板 ---
        panel = QWidget()
        panel.setFixedWidth(320)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(6)
        # 添加滚动区域
        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(panel)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll_area, stretch=1)

        tab_widget = QTabWidget()
        self.tab_widget = tab_widget  # 保存引用用于切换标签

        # Tab 1: 模型和计算
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)

        model_group = QGroupBox("模型信息")
        model_group_layout = QVBoxLayout()
        self.lbl_foot = QLabel("足部模型: 未加载")
        self.lbl_brace = QLabel("护具模型: 未加载")
        self.lbl_vertices = QLabel("顶点数: -")
        model_group_layout.addWidget(self.lbl_foot)
        model_group_layout.addWidget(self.lbl_brace)
        model_group_layout.addWidget(self.lbl_vertices)
        model_group.setLayout(model_group_layout)
        model_layout.addWidget(model_group)

        btn_load_foot = QPushButton("加载足部 STL")
        btn_load_foot.clicked.connect(self._load_foot)
        btn_load_brace = QPushButton("加载护具 STL")
        btn_load_brace.clicked.connect(self._load_brace)
        btn_pick_inner = QPushButton("选取内侧面")
        btn_pick_inner.clicked.connect(self._pick_inner_surface)
        btn_calc = QPushButton("计算距离")
        btn_calc.clicked.connect(self._compute_distance)
        btn_optimize = QPushButton("自动最优位置")
        btn_optimize.clicked.connect(self._optimize_position)
        btn_radial = QPushButton("计算径向距离")
        btn_radial.clicked.connect(self._compute_radial_distance)
        btn_reset = QPushButton("重置变换")
        btn_reset.clicked.connect(self._reset_transform)

        btn_layout = QVBoxLayout()
        btn_layout.addWidget(btn_load_foot)
        btn_layout.addWidget(btn_load_brace)
        btn_layout.addWidget(btn_pick_inner)
        btn_layout.addWidget(btn_calc)
        btn_layout.addWidget(btn_optimize)
        btn_layout.addWidget(btn_radial)
        btn_layout.addWidget(btn_reset)
        model_layout.addLayout(btn_layout)

        # 内侧面区域列表
        inner_group = QGroupBox("内侧面区域")
        inner_group_layout = QVBoxLayout()
        self.region_list = QListWidget()
        self.region_list.setSelectionMode(QListWidget.NoSelection)
        self.region_list.itemChanged.connect(self._on_region_toggled)
        self.region_list.setMaximumHeight(100)
        inner_group_layout.addWidget(self.region_list)

        # 区域操作按钮
        region_btn_layout = QHBoxLayout()
        btn_select_all = QPushButton("全选")
        btn_select_all.clicked.connect(self._select_all_regions)
        btn_deselect_all = QPushButton("全不选")
        btn_deselect_all.clicked.connect(self._deselect_all_regions)
        region_btn_layout.addWidget(btn_select_all)
        region_btn_layout.addWidget(btn_deselect_all)
        inner_group_layout.addLayout(region_btn_layout)

        inner_group.setLayout(inner_group_layout)
        model_layout.addWidget(inner_group)

        # 区域 Actor 跟踪字典：region_idx -> actor
        self._region_actor_map = {}

        # 保存位置功能
        save_group = QGroupBox("保存位置 (5 个槽位)")
        save_group_layout = QVBoxLayout()
        self.position_list = QListWidget()
        self.position_list.setMaximumHeight(120)
        self.position_list.itemDoubleClicked.connect(self._on_position_double_clicked)
        self._refresh_position_list()
        save_group_layout.addWidget(self.position_list)

        # 保存/加载按钮
        save_btn_layout = QHBoxLayout()
        btn_save = QPushButton("保存当前位置")
        btn_save.clicked.connect(self._save_position)
        btn_load = QPushButton("加载选中位置")
        btn_load.clicked.connect(self._load_position)
        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self._clear_position)
        save_btn_layout.addWidget(btn_save)
        save_btn_layout.addWidget(btn_load)
        save_btn_layout.addWidget(btn_clear)
        save_group_layout.addLayout(save_btn_layout)

        save_group.setLayout(save_group_layout)
        model_layout.addWidget(save_group)

        model_layout.addStretch()
        tab_widget.addTab(model_tab, "模型")

        # Tab 2: 颜色阈值
        color_tab = QWidget()
        color_layout = QVBoxLayout(color_tab)

        self.preset_combo = QComboBox()
        for name in PRESET_THRESHOLDS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        color_layout.addWidget(QLabel("预设方案:"))
        color_layout.addWidget(self.preset_combo)

        self.threshold_table = QTableWidget()
        self.threshold_table.setColumnCount(4)
        self.threshold_table.setHorizontalHeaderLabels(
            ["下限(mm)", "上限(mm)", "颜色", "标签"]
        )
        color_layout.addWidget(self.threshold_table)

        self._refresh_threshold_table()
        tab_widget.addTab(color_tab, "颜色")

        # Tab 3: 统计结果
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Menlo", 10))
        stats_layout.addWidget(self.stats_text)
        tab_widget.addTab(stats_tab, "统计")
        self.stats_tab_index = 2  # 记录统计标签页索引

        # Tab 4: 操作设置
        control_tab = QWidget()
        control_layout = QVBoxLayout(control_tab)

        control_layout.addWidget(QLabel("移动步长 (mm):"))
        self.spin_move = QDoubleSpinBox()
        self.spin_move.setRange(0.01, 10.0)
        self.spin_move.setSingleStep(0.1)
        self.spin_move.setValue(self.move_step)
        self.spin_move.valueChanged.connect(self._on_move_step_changed)
        control_layout.addWidget(self.spin_move)

        control_layout.addWidget(QLabel("旋转步长 (度):"))
        self.spin_rotate = QDoubleSpinBox()
        self.spin_rotate.setRange(0.01, 10.0)
        self.spin_rotate.setSingleStep(0.1)
        self.spin_rotate.setValue(self.rotate_step)
        self.spin_rotate.valueChanged.connect(self._on_rotate_step_changed)
        control_layout.addWidget(self.spin_rotate)

        self.btn_fine = QPushButton("进入精细调节模式")
        self.btn_fine.setCheckable(True)
        self.btn_fine.clicked.connect(self._toggle_fine_mode)
        control_layout.addWidget(self.btn_fine)

        self.btn_wireframe = QPushButton("切换线框/实体")
        self.btn_wireframe.clicked.connect(self._toggle_wireframe)
        control_layout.addWidget(self.btn_wireframe)

        self.btn_measure = QPushButton("测量距离")
        self.btn_measure.setCheckable(True)
        self.btn_measure.clicked.connect(self._toggle_measurement)
        control_layout.addWidget(self.btn_measure)

        control_layout.addStretch()
        tab_widget.addTab(control_tab, "设置")

        # Tab 5: 护具尺寸修改
        deform_tab = QWidget()
        deform_layout = QVBoxLayout(deform_tab)

        # STEP 加载/转换区域
        step_load_group = QGroupBox("STEP 加载")
        step_load_layout = QVBoxLayout()

        btn_load_step = QPushButton("加载护具 STEP 文件")
        btn_load_step.clicked.connect(self._load_brace_step)
        step_load_layout.addWidget(btn_load_step)

        btn_convert_brace = QPushButton("将已加载护具转换为 STEP")
        btn_convert_brace.clicked.connect(self._convert_brace_to_step)
        step_load_layout.addWidget(btn_convert_brace)

        step_load_group.setLayout(step_load_layout)
        deform_layout.addWidget(step_load_group)

        # 变形区域选择提示
        self.lbl_deform_region = QLabel("变形区域：请先在「模型」标签页选取内侧面并勾选区域")
        self.lbl_deform_region.setWordWrap(True)
        deform_layout.addWidget(self.lbl_deform_region)

        # 变形方向和偏移量
        direction_group = QGroupBox("变形方向")
        direction_layout = QHBoxLayout()

        self.btn_shrink = QPushButton("向内收缩")
        self.btn_shrink.setCheckable(True)
        self.btn_shrink.setChecked(True)
        self.btn_shrink.clicked.connect(lambda: self._set_deform_direction("shrink"))

        self.btn_expand = QPushButton("向外扩张")
        self.btn_expand.setCheckable(True)
        self.btn_expand.clicked.connect(lambda: self._set_deform_direction("expand"))

        direction_layout.addWidget(self.btn_shrink)
        direction_layout.addWidget(self.btn_expand)
        direction_group.setLayout(direction_layout)
        deform_layout.addWidget(direction_group)

        # 偏移量
        deform_layout.addWidget(QLabel("偏移量 (mm):"))
        self.spin_deform_offset = QDoubleSpinBox()
        self.spin_deform_offset.setRange(0.1, 20.0)
        self.spin_deform_offset.setSingleStep(0.5)
        self.spin_deform_offset.setValue(5.0)
        deform_layout.addWidget(self.spin_deform_offset)

        # 衰减半径
        deform_layout.addWidget(QLabel("衰减半径 (mm, 0=无衰减):"))
        self.spin_deform_decay = QDoubleSpinBox()
        self.spin_deform_decay.setRange(0.0, 100.0)
        self.spin_deform_decay.setSingleStep(1.0)
        self.spin_deform_decay.setValue(0.0)
        deform_layout.addWidget(self.spin_deform_decay)

        # 边界平滑半径
        self.spin_boundary_smooth = QDoubleSpinBox()
        self.spin_boundary_smooth.setRange(0.0, 50.0)
        self.spin_boundary_smooth.setSingleStep(1.0)
        self.spin_boundary_smooth.setValue(5.0)
        deform_layout.addWidget(QLabel("边界平滑 (mm, 0=不平滑):"))
        deform_layout.addWidget(self.spin_boundary_smooth)

        # 操作按钮
        deform_btn_row = QHBoxLayout()
        self.btn_deform_preview = QPushButton("预览")
        self.btn_deform_preview.clicked.connect(self._preview_deformation)
        self.btn_deform_apply = QPushButton("应用")
        self.btn_deform_apply.clicked.connect(self._apply_deformation)
        self.btn_deform_revert = QPushButton("还原")
        self.btn_deform_revert.clicked.connect(self._revert_preview)
        self.btn_deform_undo = QPushButton("撤销")
        self.btn_deform_undo.clicked.connect(self._undo_deformation)
        deform_btn_row.addWidget(self.btn_deform_preview)
        deform_btn_row.addWidget(self.btn_deform_apply)
        deform_btn_row.addWidget(self.btn_deform_revert)
        deform_btn_row.addWidget(self.btn_deform_undo)
        deform_layout.addLayout(deform_btn_row)

        # 导出按钮
        export_btn_row = QHBoxLayout()
        self.btn_export_stl = QPushButton("导出 STL")
        self.btn_export_stl.clicked.connect(self._export_stl)
        self.btn_export_step = QPushButton("导出 STEP")
        self.btn_export_step.clicked.connect(self._export_step)
        export_btn_row.addWidget(self.btn_export_stl)
        export_btn_row.addWidget(self.btn_export_step)
        deform_layout.addLayout(export_btn_row)

        deform_layout.addStretch()
        tab_widget.addTab(deform_tab, "尺寸修改")

        panel_layout.addWidget(tab_widget)

        # 坐标信息
        coord_group = QGroupBox("护具位置")
        coord_layout = QVBoxLayout()
        self.lbl_dx = QLabel("ΔX = 0.00 mm")
        self.lbl_dy = QLabel("ΔY = 0.00 mm")
        self.lbl_dz = QLabel("ΔZ = 0.00 mm")
        coord_layout.addWidget(self.lbl_dx)
        coord_layout.addWidget(self.lbl_dy)
        coord_layout.addWidget(self.lbl_dz)

        # 旋转角度显示
        self.lbl_rx = QLabel("Rx = 0.0°")
        self.lbl_ry = QLabel("Ry = 0.0°")
        self.lbl_rz = QLabel("Rz = 0.0°")
        coord_layout.addWidget(self.lbl_rx)
        coord_layout.addWidget(self.lbl_ry)
        coord_layout.addWidget(self.lbl_rz)

        coord_group.setLayout(coord_layout)
        panel_layout.addWidget(coord_group)


        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

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

    # ================================================================
    # VTK 设置
    # ================================================================

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

                # ---- 方向键：相机操作 ----
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

    # ================================================================
    # 功能方法
    # ================================================================

    def _load_foot(self):
        """加载足部 STL 文件"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择足部 STL 文件", "", "STL Files (*.stl)"
        )
        if not filepath:
            return
        try:
            self.foot_model = load_stl(filepath)
            self.lbl_foot.setText(f"足部模型: {self.foot_model.name}")

            # 清除旧的距离标示
            self.scene.clear_min_max_indicators()

            # 添加到场景
            if self.scene._foot_actor:
                self.scene.renderer.RemoveActor(self.scene._foot_actor)
            self.scene.add_foot_model(self.foot_model.polydata)

            # 初始化距离计算器
            self.distance_calc = DistanceCalculator(
                self.foot_model.polydata
            )

            # 两个模型都加载后，调整坐标轴
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
        """加载护具 STL 文件（直接加载，不转换 STEP）"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择护具 STL 文件", "", "STL Files (*.stl)"
        )
        if not filepath:
            return

        self._brace_step_filepath = None
        self._prepare_and_load(filepath)

    def _prepare_and_load(self, stl_path: str):
        """清除旧状态并加载护具 STL（通用流程）"""
        try:
            # 清除旧的内侧面
            self._refresh_inner_highlight()
            self._inner_cells = None
            self._inner_regions = []
            self._inner_vertex_indices = None
            self._region_actor_map.clear()
            self.region_list.clear()

            # 清除旧的距离标示和变形状态
            self.scene.clear_min_max_indicators()
            self.deformation_state.clear()
            self.deformation_engine = None
            self._deformed_inner_vertices = None
            self._original_inner_vertices = None

            self.brace_model = load_stl(stl_path)
            self.lbl_brace.setText(f"护具模型: {self.brace_model.name}")
            self.brace_transform = TransformManager(
                center=self.brace_model.centroid
            )

            # 添加到场景
            if self.scene._brace_actor:
                self.scene.renderer.RemoveActor(self.scene._brace_actor)
            self.scene.add_brace_model(self.brace_model.polydata)

            # 两个模型都加载后，调整坐标轴
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

            # 保存当前护具文件路径用于位置文件管理
            self._current_brace_filepath = stl_path

            # 自动加载保存的位置
            self._auto_load_brace_positions(stl_path)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法加载护具模型:\n{e}")

    def _convert_brace_to_step(self):
        """将当前已加载的护具 STL 转换为 STEP 进行高精度处理"""
        if self.brace_model is None:
            QMessageBox.warning(self, "警告", "请先在「模型」标签页加载护具 STL")
            return

        # 检查是否已有对应的 STEP 文件
        brace_path = self._current_brace_filepath
        if brace_path and os.path.exists(brace_path):
            # 尝试同目录下同名的 STEP 文件
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

        # 执行转换：当前护具 STL → STEP → 高精度 STL
        try:
            self._step_progress = QProgressDialog(
                "正在转换为 STEP 并网格化...",
                "取消", 0, 0, self
            )
            self._step_progress.setWindowTitle("STEP 转换")
            self._step_progress.setWindowModality(Qt.WindowModal)
            self._step_progress.setMinimumDuration(0)
            self._step_progress.show()

            # 临时 STL 文件
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
        """当前护具转换为 STEP 完成回调"""
        self._step_progress.close()
        self._brace_step_filepath = stl_path
        self._prepare_and_load(stl_path)
        self.status_bar.showMessage(
            f"已加载护具 (STEP 高精度): {Path(stl_path).name}"
        )

    def _load_brace_step_from_file(self, step_path: str):
        """从指定 STEP 文件加载护具"""
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

    def _pick_inner_surface(self):
        """自动识别护具内侧面，拆分为连通区域"""
        if self.brace_model is None:
            QMessageBox.warning(self, "警告", "请先加载护具模型")
            return

        self.status_bar.showMessage("正在识别内侧面...")

        # 自动识别内侧面
        self._inner_cells = identify_inner_surface(
            self.brace_model.polydata
        )
        self._inner_vertex_indices = extract_inner_vertices(
            self.brace_model.polydata, self._inner_cells
        )

        # 拆分为连通区域
        # 使用较大的角度阈值和最小区域大小，减少区域数量
        self._inner_regions = find_connected_regions(
            self._inner_cells,
            self.brace_model.polydata,
            angle_threshold=75.0,  # 较大的角度阈值，使更多面被归为同一区域
            min_region_size=150,   # 最小区域大小，过滤小碎片
        )

        # 高亮显示各区域
        self._refresh_inner_highlight()
        self._render()

        # 更新区域列表
        self.region_list.clear()
        # 使用高饱和度、高对比度的颜色方案，确保各区域清晰可辨
        colors = [
            (255, 50, 50),    # 区域 1: 鲜艳红色
            (50, 200, 50),    # 区域 2: 鲜亮绿色
            (50, 100, 255),   # 区域 3: 纯蓝色
            (255, 140, 50),   # 区域 4: 鲜艳橙色
            (200, 50, 255),   # 区域 5: 鲜艳紫色
            (50, 255, 255),   # 区域 6: 纯青色
            (255, 100, 180),  # 区域 7: 艳粉色
            (150, 255, 100),  # 区域 8: 黄绿色
            (100, 180, 255),  # 区域 9: 天蓝色
            (255, 200, 100),  # 区域 10: 金黄色
        ]
        for i, region in enumerate(self._inner_regions):
            color = colors[i % len(colors)]
            item = QListWidgetItem(f"区域 {i+1} ({len(region):,} 面)")
            item.setCheckState(Qt.Unchecked)  # 默认不选
            item.setData(Qt.UserRole, (i, color))
            self.region_list.addItem(item)

        n_inner = len(self._inner_cells)
        n_vertices = len(self._inner_vertex_indices)
        n_total = self.brace_model.triangle_count
        self.status_bar.showMessage(
            f"已选取内侧面: {n_inner:,} 三角面 / {n_total:,} 总面 "
            f"({n_inner/n_total*100:.1f}%), {n_vertices:,} 顶点, "
            f"{len(self._inner_regions)} 个区域 (已过滤小于 150 面的碎片)"
        )

    def _on_region_toggled(self, item: QListWidgetItem):
        """区域 checkbox 状态变化 - 增量更新高亮显示"""
        # 获取当前 item 对应的区域索引
        region_idx, color = item.data(Qt.UserRole)
        region_cells = self._inner_regions[region_idx]

        # 淡蓝色高亮颜色 (100, 180, 255) - 所有勾选的区域都使用统一的淡蓝色
        highlight_color = (100, 180, 255)

        if item.checkState() == Qt.Checked:
            # 勾选：添加该区域的高亮 Actor（使用大红色）
            if region_idx not in self._region_actor_map:
                actor = self.scene._create_inner_surface_actor(
                    self.brace_model.polydata, region_cells, highlight_color
                )
                self.scene.renderer.AddActor(actor)
                self._region_actor_map[region_idx] = actor
        else:
            # 取消勾选：移除该区域的高亮 Actor
            if region_idx in self._region_actor_map:
                actor = self._region_actor_map[region_idx]
                self.scene.renderer.RemoveActor(actor)
                del self._region_actor_map[region_idx]

        self._render()

    def _select_all_regions(self):
        """全选所有区域"""
        for i in range(self.region_list.count()):
            item = self.region_list.item(i)
            item.setCheckState(Qt.Checked)

    def _deselect_all_regions(self):
        """全不选所有区域"""
        for i in range(self.region_list.count()):
            item = self.region_list.item(i)
            item.setCheckState(Qt.Unchecked)

    def _refresh_inner_highlight(self):
        """清除旧的内侧面高亮"""
        for actor in self._region_actor_map.values():
            self.scene.renderer.RemoveActor(actor)
        self._region_actor_map.clear()
        for label in self._region_labels:
            self.scene.renderer.RemoveActor(label)
        self._region_labels = []
        self.scene.clear_inner_surface_actors()

    def _get_selected_inner_vertices(self) -> Optional[np.ndarray]:
        """
        获取用户勾选的内侧面区域对应的顶点

        返回:
            变换后的顶点数组 (N, 3)，如果没有勾选任何区域则返回 None
        """
        if not self._inner_regions or self._inner_vertex_indices is None:
            return None

        # 收集所有勾选区域的 cell 索引
        selected_cells = set()
        for i in range(self.region_list.count()):
            item = self.region_list.item(i)
            if item.checkState() == Qt.Checked:
                region_idx, _ = item.data(Qt.UserRole)
                selected_cells.update(self._inner_regions[region_idx].tolist())

        if not selected_cells:
            return None

        # 从选中的 cell 中提取顶点索引
        selected_vertex_set = set()
        for cell_id in selected_cells:
            cell = self.brace_model.polydata.GetCell(cell_id)
            for j in range(cell.GetNumberOfPoints()):
                selected_vertex_set.add(cell.GetPointId(j))

        selected_indices = np.array(sorted(selected_vertex_set), dtype=np.int64)
        self._current_selected_indices = selected_indices

        # 应用当前变换
        return get_transformed_vertices(
            self.brace_model.polydata,
            selected_indices,
            self.brace_transform.get_matrix(),
        )

    def _compute_distance(self):
        """计算距离并着色"""
        if self.foot_model is None:
            QMessageBox.warning(self, "警告", "请先加载足部模型")
            return
        if self.brace_model is None:
            QMessageBox.warning(self, "警告", "请先加载护具模型")
            return
        if self.distance_calc is None:
            QMessageBox.warning(self, "警告", "足部模型未正确初始化")
            return

        self.status_bar.showMessage("正在计算距离...")
        self._render()  # 刷新UI显示进度

        # 获取用户确认的内侧面顶点（只计算勾选的区域）
        selected_vertices = self._get_selected_inner_vertices()

        if selected_vertices is None or len(selected_vertices) == 0:
            # 如果没有勾选任何区域，使用全部内侧面或全部顶点
            if self._inner_vertex_indices is not None:
                selected_vertices = get_transformed_vertices(
                    self.brace_model.polydata,
                    self._inner_vertex_indices,
                    self.brace_transform.get_matrix(),
                )
                self._current_selected_indices = self._inner_vertex_indices
            else:
                original_vertices = _polydata_to_vertices(self.brace_model.polydata)
                selected_vertices = self.brace_transform.apply(original_vertices)
                self._current_selected_indices = None

        # 计算有符号距离
        stats = self.distance_calc.compute_with_stats(
            selected_vertices, self.current_thresholds
        )
        self.current_distances = stats.distances

        # 应用颜色：如果是内侧面，只着色内侧面顶点
        if self._current_selected_indices is not None:
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

        # 记录位置用于增量判断
        self._last_calc_position = self.brace_transform.get_translation()

        # 更新统计信息
        self._update_stats(stats)

        # 标示最短/最长距离点
        self._draw_min_max_indicators(selected_vertices)

        self._render()

        self.status_bar.showMessage(
            f"计算完成 | 最小: {stats.min_dist:+.2f}mm | "
            f"平均: {stats.mean_dist:+.2f}mm | "
            f"穿透: {stats.penetration_count} 点"
        )

    def _draw_min_max_indicators(self, selected_vertices: np.ndarray):
        """在场景中标示最短/最长距离点"""
        if self.current_distances is None or len(self.current_distances) == 0:
            return
        if self.distance_calc is None:
            return

        min_idx = int(np.argmin(self.current_distances))
        max_idx = int(np.argmax(self.current_distances))
        min_pos = selected_vertices[min_idx]
        max_pos = selected_vertices[max_idx]
        min_val = float(self.current_distances[min_idx])
        max_val = float(self.current_distances[max_idx])

        # 查询足部表面上的最近点
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

    # ---- 护具尺寸修改 ----

    def _set_deform_direction(self, direction: str):
        """设置变形方向"""
        self._deform_direction = direction
        self.btn_shrink.setChecked(direction == "shrink")
        self.btn_expand.setChecked(direction == "expand")

    def _load_brace_step(self):
        """从 STEP 文件加载护具"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择护具 STEP 文件", "", "STEP Files (*.step *.stp)"
        )
        if not filepath:
            return
        self._load_brace_step_from_file(filepath)

    def _on_step_conversion_done(self, stl_path: str):
        """STEP 转换完成回调"""
        self._step_progress.close()

        # 清除旧状态
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

        self._prepare_and_load(stl_path)

        self.status_bar.showMessage(
            f"已加载护具 STEP: {Path(self._brace_step_filepath).name} "
            f"(→ {Path(stl_path).name})"
        )

    def _on_step_conversion_error(self, error_msg: str):
        """STEP 转换错误回调"""
        self._step_progress.close()
        QMessageBox.critical(self, "加载失败", f"无法加载 STEP 文件:\n{error_msg}")
        self.status_bar.showMessage("STEP 加载失败")

    def _ensure_deformation_engine(self):
        """确保变形引擎已初始化"""
        if (self.deformation_engine is None
                and self.foot_model is not None
                and self.brace_model is not None
                and self._inner_vertex_indices is not None):
            self.deformation_engine = DeformationEngine(
                self.foot_model.polydata,
                self.brace_model.polydata,
                self._inner_vertex_indices,
            )
            # 保存原始顶点位置
            pts = self.brace_model.polydata.GetPoints()
            self._original_inner_vertices = np.array([
                pts.GetPoint(i) for i in self._inner_vertex_indices
            ])
            self._deformed_inner_vertices = self._original_inner_vertices.copy()

    def _get_deformation_params(self) -> DeformationParams:
        """从 UI 获取当前变形参数（基于用户勾选的区域）"""
        selected_vertex_indices = self._get_selected_region_indices()

        if selected_vertex_indices is None or len(selected_vertex_indices) == 0:
            QMessageBox.warning(
                self, "警告",
                "请先在「模型」标签页的「内侧面区域」列表中勾选要变形的区域"
            )
            return None

        offset = self.spin_deform_offset.value()
        if self._deform_direction == "expand":
            offset = -offset

        return DeformationParams(
            mode="normal",
            region_indices=selected_vertex_indices,
            offset_mm=offset,
            scale_factor=1.0 + offset / 100.0,
            decay_radius=self.spin_deform_decay.value(),
            boundary_smooth=self.spin_boundary_smooth.value(),
        )

    def _get_selected_region_indices(self) -> Optional[np.ndarray]:
        """获取用户勾选区域的顶点索引（相对于内表面数组的 0..N-1 索引）"""
        if not self._inner_regions or self._inner_vertex_indices is None:
            return None

        # 收集所有勾选区域的 cell 索引
        selected_cells = set()
        for i in range(self.region_list.count()):
            item = self.region_list.item(i)
            if item.checkState() == Qt.Checked:
                region_idx, _ = item.data(Qt.UserRole)
                selected_cells.update(self._inner_regions[region_idx].tolist())

        if not selected_cells:
            return None

        # 从选中的 cell 中提取顶点索引
        selected_vertex_set = set()
        for cell_id in selected_cells:
            cell = self.brace_model.polydata.GetCell(cell_id)
            for j in range(cell.GetNumberOfPoints()):
                selected_vertex_set.add(cell.GetPointId(j))

        # 将原始网格索引映射到内表面数组索引（0..N-1）
        index_map = {idx: i for i, idx in enumerate(self._inner_vertex_indices)}
        mapped_indices = np.array(
            sorted(index_map[v] for v in selected_vertex_set if v in index_map),
            dtype=np.int64,
        )
        return mapped_indices if len(mapped_indices) > 0 else None

    def _preview_deformation(self):
        """预览变形效果（不提交到历史）"""
        if self.brace_model is None or self._inner_vertex_indices is None:
            QMessageBox.warning(self, "警告", "请先加载护具并在「模型」标签页选取内侧面")
            return

        self._ensure_deformation_engine()
        if self.deformation_engine is None:
            QMessageBox.warning(self, "警告", "变形引擎初始化失败")
            return

        params = self._get_deformation_params()
        if params is None:
            return

        deformed = self.deformation_engine.apply(
            self._deformed_inner_vertices, params
        )
        self._preview_vertices = deformed

        self._apply_vertex_update(deformed)
        self._render()

        direction_label = "向内收缩" if params.offset_mm > 0 else "向外扩张"
        self.status_bar.showMessage(
            f"预览: {direction_label} {abs(params.offset_mm):.1f}mm — "
            f"满意后点击「应用」提交，或点击「还原」取消"
        )

    def _apply_deformation(self):
        """提交预览变形到历史，或直接应用"""
        if self.brace_model is None or self._inner_vertex_indices is None:
            QMessageBox.warning(self, "警告", "请先加载护具并在「模型」标签页选取内侧面")
            return

        self._ensure_deformation_engine()
        if self.deformation_engine is None:
            return

        if self._preview_vertices is not None:
            final = self._preview_vertices
            params = self._get_deformation_params()
        else:
            params = self._get_deformation_params()
            if params is None:
                return
            final = self.deformation_engine.apply(
                self._deformed_inner_vertices, params
            )

        step = DeformationStep(
            mode=params.mode,
            offset_mm=params.offset_mm,
            scale_factor=params.scale_factor,
            decay_radius=params.decay_radius,
            direction=(
                params.direction.tolist() if params.direction is not None else None
            ),
            center_point=(
                params.center_point.tolist() if params.center_point is not None else None
            ),
            region_indices=params.region_indices.tolist(),
        )
        self.deformation_state.push(step)
        self._deformed_inner_vertices = final.copy()
        self._preview_vertices = None

        self._render()

        if self.current_distances is not None:
            self._compute_distance()

        direction_label = "向内收缩" if params.offset_mm > 0 else "向外扩张"
        n_regions = len(params.region_indices)
        self.status_bar.showMessage(
            f"已应用{direction_label} {abs(params.offset_mm):.1f}mm "
            f"(区域 {n_regions:,} 顶点, 共 {self.deformation_state.step_count} 步)"
        )

    def _revert_preview(self):
        """还原预览，回到变形前的状态"""
        if self._preview_vertices is None and self.deformation_state.step_count == 0:
            self.status_bar.showMessage("当前没有可还原的操作")
            return

        self._preview_vertices = None

        # 重新应用所有已提交的变形步骤
        self._replay_deformations()
        self._render()
        self.status_bar.showMessage("已还原预览，回到变形前状态")

    def _undo_deformation(self):
        """撤销上一步变形"""
        if self.deformation_state.step_count == 0:
            self.status_bar.showMessage("没有可撤销的操作")
            return

        self.deformation_state.undo()

        # 从头重新应用所有变形
        self._replay_deformations()
        self._render()
        self.status_bar.showMessage(
            f"已撤销 (剩余 {self.deformation_state.step_count} 步)"
        )

    def _replay_deformations(self):
        """从头重新应用所有变形步骤"""
        if self._original_inner_vertices is None:
            return
        if self.deformation_engine is None:
            return

        self._deformed_inner_vertices = self._original_inner_vertices.copy()
        for params in self.deformation_state.get_params_list():
            self._deformed_inner_vertices = self.deformation_engine.apply(
                self._deformed_inner_vertices, params
            )
        self._apply_vertex_update(self._deformed_inner_vertices)

    def _apply_vertex_update(self, deformed_vertices: np.ndarray):
        """将变形后的顶点更新到 VTK polydata 并刷新场景"""
        if self.brace_model is None:
            return

        pts = self.brace_model.polydata.GetPoints()
        for i, idx in enumerate(self._inner_vertex_indices):
            pts.SetPoint(int(idx), deformed_vertices[i].tolist())
        pts.Modified()
        self.brace_model.polydata.Modified()

    def _export_stl(self):
        """导出当前护具为 STL"""
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
        """导出当前护具为 STEP（通过 STL → STEP 转换）"""
        if self.brace_model is None:
            QMessageBox.warning(self, "警告", "请先加载护具")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出 STEP", "", "STEP Files (*.step *.stp)"
        )
        if not filepath:
            return

        try:
            # 先导出为临时 STL
            temp_stl = filepath + ".temp.stl"
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(temp_stl)
            writer.SetInputData(self.brace_model.polydata)
            writer.Write()

            # STL → STEP
            stl_to_step(temp_stl, filepath, tolerance=0.1)

            # 清理临时文件
            if os.path.exists(temp_stl):
                os.unlink(temp_stl)

            self.status_bar.showMessage(f"已导出 STEP: {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"无法导出 STEP:\n{e}")

    def _reset_transform(self):
        """重置护具变换"""
        self.brace_transform.reset()
        self._update_brace_view()
        self.scene.clear_distance_colors()
        self.scene.clear_min_max_indicators()
        self.current_distances = None
        self.stats_text.clear()

        # 清除变形状态
        self.deformation_state.clear()
        self._deformed_inner_vertices = None
        self._original_inner_vertices = None

        # 清除内侧面高亮和状态
        self._refresh_inner_highlight()
        self._inner_cells = None
        self._inner_regions = []
        self._inner_vertex_indices = None
        self._region_actor_map.clear()
        self.region_list.clear()

        self._render()

    def _reset_camera(self):
        """重置相机"""
        self.scene.reset_camera()
        self._render()

    def _update_brace_view(self):
        """更新护具显示并触发实时计算"""
        if self.scene._brace_actor is None:
            return
        self.scene.apply_transform(self.brace_transform.get_matrix())
        self._update_coord_display()

        # 节流重新计算
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
                    QTimer.singleShot(
                        200, self._debounced_recompute
                    )

        self._render()

    def _debounced_recompute(self):
        """节流后的重新计算"""
        self._pending_recalc = False
        self._compute_distance()

    def _optimize_position(self):
        """自动计算最优佩戴位置"""
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

        # 询问用户是否使用当前勾选的区域
        selected_vertices = self._get_selected_inner_vertices()
        if selected_vertices is None or len(selected_vertices) == 0:
            # 使用全部内侧面
            indices_to_use = self._inner_vertex_indices
        else:
            indices_to_use = self._current_selected_indices

        self.status_bar.showMessage("正在计算最优位置（粗 + 精两级搜索）...")
        self._render()

        # 创建优化器
        optimizer = BraceOptimizer(
            self.foot_model.polydata,
            self.brace_model.polydata,
            indices_to_use,
        )

        # 执行优化（基于当前位置，范围±10mm，旋转±5 度）
        current_translation = self.brace_transform.get_translation()
        current_rotation = self.brace_transform.get_rotation()

        result = optimizer.optimize(
            current_translation=current_translation,
            current_rotation=current_rotation,
            search_range=10.0,     # ±10mm
            search_step=2.0,       # 2mm 步长
            rotation_range=5.0,    # ±5 度
            rotation_step=2.5,     # 2.5 度步长
        )

        # 应用最优位置
        self.brace_transform.reset()
        self.brace_transform.translate(*result.translation.tolist())
        if np.any(result.rotation):
            self.brace_transform.rotate_x(result.rotation[0])
            self.brace_transform.rotate_y(result.rotation[1])
            self.brace_transform.rotate_z(result.rotation[2])

        self._update_brace_view()

        # 重新计算距离
        self._compute_distance()

        # 显示结果
        result_text = (
            f"{'='*35}\n"
            f" 优化结果（径向距离）\n"
            f"{'='*35}\n\n"
            f"平移：X={result.translation[0]:+.2f}, "
            f"Y={result.translation[1]:+.2f}, "
            f"Z={result.translation[2]:+.2f} mm\n\n"
            f"旋转：RX={result.rotation[0]:+.1f}, "
            f"RY={result.rotation[1]:+.1f}, "
            f"RZ={result.rotation[2]:+.1f} 度\n\n"
            f"理想区间 (4-6mm) 覆盖率：{result.coverage_4_6mm:.1f}%\n"
            f"理想区间点数：{result.ideal_count} / {result.total_count}\n\n"
            f"平均径向间隙：{result.mean_distance:.2f} mm\n"
            f"间隙标准差：{result.std_distance:.2f} mm\n"
            f"{'='*35}"
        )

        self.stats_text.setText(result_text)
        self.status_bar.showMessage(
            f"优化完成 | 理想覆盖率：{result.coverage_4_6mm:.1f}% | "
            f"平均径向间隙：{result.mean_distance:.2f}mm"
        )

        # 弹窗提示
        QMessageBox.information(
            self, "优化完成",
            f"最优位置计算完成！\n\n"
            f"理想区间 (4-6mm) 覆盖率：{result.coverage_4_6mm:.1f}%\n"
            f"平均径向间隙：{result.mean_distance:.2f} mm\n\n"
            f"详细结果已显示在'统计'标签页"
        )

    def _compute_radial_distance(self):
        """基于中轴线的径向距离计算"""
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
            # 获取用户确认的内侧面顶点（只计算勾选的区域）
            selected_vertices = self._get_selected_inner_vertices()

            if selected_vertices is None or len(selected_vertices) == 0:
                # 使用全部内侧面
                indices_to_use = self._inner_vertex_indices
                selected_vertices = get_transformed_vertices(
                    self.brace_model.polydata,
                    self._inner_vertex_indices,
                    self.brace_transform.get_matrix(),
                )
            else:
                indices_to_use = self._current_selected_indices

            # 创建变换后的护具 polydata 用于轴线计算
            transformed_brace = vtk.vtkTransformPolyDataFilter()
            transformed_brace.SetInputData(self.brace_model.polydata)
            transform = vtk.vtkTransform()
            transform.SetMatrix(self.brace_transform.get_matrix().flatten())
            transformed_brace.SetTransform(transform)
            transformed_brace.Update()

            # 创建径向距离计算器（使用变换后的护具计算轴线）
            self.radial_calc = RadialDistanceCalculator(
                foot_polydata=self.foot_model.polydata,
                brace_polydata=transformed_brace.GetOutput(),
                inner_vertex_indices=indices_to_use,
                axis_direction=np.array([0.0, 0.0, 1.0]),  # Z 轴
            )

            # 计算径向距离
            stats = self.radial_calc.compute_radial_distances(selected_vertices)

            # 记录位置用于增量判断
            self._last_calc_position = self.brace_transform.get_translation()

            # 应用颜色显示（使用与计算距离相同的颜色方案）
            self.current_distances = stats.radial_distances
            self._current_selected_indices = indices_to_use

            # 应用颜色到内侧面
            self.scene.apply_inner_surface_colors(
                self.brace_model.polydata,
                self._current_selected_indices,
                self.current_distances,
                self.current_thresholds,
            )

            # 更新统计信息
            self._update_radial_stats(stats)

            # 标示最短/最长距离点
            self._draw_min_max_indicators(selected_vertices)

            # 切换到统计标签页
            self.tab_widget.setCurrentIndex(self.stats_tab_index)

            self._render()

            self.status_bar.showMessage(
                f"径向距离计算完成 | 平均间隙：{stats.mean_gap:.2f}mm | "
                f"理想区间 (4-6mm): {stats.ideal_4_6_ratio:.1f}%"
            )

        except Exception as e:
            import traceback
            error_msg = f"计算失败：{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "错误", error_msg)
            self.status_bar.showMessage("计算失败，详见错误弹窗")

    def _update_radial_stats(self, stats):
        """更新径向距离统计信息显示"""
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

        # 距离分布直方图
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

    def _update_coord_display(self):
        """更新坐标显示"""
        t = self.brace_transform.get_translation()
        r = self.brace_transform.get_rotation()
        self.lbl_dx.setText(f"ΔX = {t[0]:+.2f} mm")
        self.lbl_dy.setText(f"ΔY = {t[1]:+.2f} mm")
        self.lbl_dz.setText(f"ΔZ = {t[2]:+.2f} mm")
        self.lbl_rx.setText(f"Rx = {r[0]:+.1f}°")
        self.lbl_ry.setText(f"Ry = {r[1]:+.1f}°")
        self.lbl_rz.setText(f"Rz = {r[2]:+.1f}°")

        # 3D 视图中的文字
        coord_text = (
            f"护具位置：ΔX={t[0]:+.2f}  ΔY={t[1]:+.2f}  ΔZ={t[2]:+.2f} mm\n"
            f"旋转角度：Rx={r[0]:+.1f}° Ry={r[1]:+.1f}° Rz={r[2]:+.1f}°"
        )
        self.scene.add_coord_text(coord_text)

    def _update_axes(self):
        """更新坐标轴（基于模型包围盒）"""
        # 获取两个模型的包围盒
        foot_bounds = self.foot_model.polydata.GetBounds()
        brace_bounds = self.brace_model.polydata.GetBounds()

        # 合并包围盒
        combined_bounds = (
            min(foot_bounds[0], brace_bounds[0]),
            max(foot_bounds[1], brace_bounds[1]),
            min(foot_bounds[2], brace_bounds[2]),
            max(foot_bounds[3], brace_bounds[3]),
            min(foot_bounds[4], brace_bounds[4]),
            max(foot_bounds[5], brace_bounds[5]),
        )

        # 更新坐标轴
        self.scene.add_axes(combined_bounds)

    def _update_model_info(self):
        """更新顶点数显示"""
        parts = []
        if self.foot_model:
            parts.append(f"足部: {self.foot_model.vertex_count:,} 顶点")
        if self.brace_model:
            parts.append(f"护具: {self.brace_model.vertex_count:,} 顶点")
        if parts:
            self.lbl_vertices.setText(" | ".join(parts))

    def _update_stats(self, stats):
        """更新统计信息面板"""
        distances = stats.distances
        total = len(distances)

        # 统计各区间点数和百分比
        penetration = int(np.sum(distances < 0))
        contact = int(np.sum((distances >= 0) & (distances < 4)))
        ideal = int(np.sum((distances >= 4) & (distances <= 6)))
        loose = int(np.sum((distances > 6) & (distances < 10)))
        too_loose = int(np.sum(distances >= 10))

        text = (
            f"{'='*35}\n"
            f" 距离统计 (单位：mm)\n"
            f"{'='*35}\n"
            f" 最小距离：  {stats.min_dist:+.2f}\n"
            f" 最大距离：  {stats.max_dist:+.2f}\n"
            f" 平均距离：  {stats.mean_dist:+.2f}\n"
            f" 中位数：    {stats.median_dist:+.2f}\n"
            f" 标准差：    {stats.std_dist:.2f}\n"
            f"{'='*35}\n"
            f" 距离分布统计:\n"
            f"  穿透 (<0):      {penetration:5d} 点 ({penetration/total*100:5.1f}%)\n"
            f"  偏紧 (0-4):     {contact:5d} 点 ({contact/total*100:5.1f}%)\n"
            f"  理想 (4-6):     {ideal:5d} 点 ({ideal/total*100:5.1f}%)\n"
            f"  偏松 (6-10):    {loose:5d} 点 ({loose/total*100:5.1f}%)\n"
            f"  过松 (>10):     {too_loose:5d} 点 ({too_loose/total*100:5.1f}%)\n"
            f"{'='*35}\n"
            f" 总顶点数：  {total}\n"
            f" 理想覆盖率：{ideal/total*100:.1f}%\n"
            f"{'='*35}"
        )
        self.stats_text.setText(text)


    def _on_preset_changed(self, preset_name: str):
        """预设方案切换"""
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
        """刷新阈值表格"""
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

    def _on_move_step_changed(self, value: float):
        self.move_step = value
        if self.is_fine_mode:
            self.move_step = max(value, 0.01)

    def _on_rotate_step_changed(self, value: float):
        self.rotate_step = value

    def _toggle_fine_mode(self, checked: bool):
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


    def _auto_load_brace_positions(self, filepath: str):
        """自动加载护具位置文件"""
        result = self.position_file_manager.load_positions(filepath)

        if result["found"]:
            self.saved_positions = result["positions"]
            self._refresh_position_list()

            # 恢复内侧面信息
            if result.get("inner_surface_indices") is not None:
                self._inner_vertex_indices = result["inner_surface_indices"]
            if result.get("inner_regions") is not None:
                self._inner_regions = result["inner_regions"]

            count = sum(1 for p in self.saved_positions if p is not None)
            self.status_bar.showMessage(f"已自动加载 {count} 个保存的位置")
        else:
            # 没有保存的位置文件，重置为默认值
            self.saved_positions = [None] * 5
            self._refresh_position_list()

    def _refresh_position_list(self):
        """刷新位置列表显示"""
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
        """保存当前位置到选中的槽位"""
        if self.brace_model is None:
            QMessageBox.warning(self, "警告", "请先加载护具模型")
            return

        # 获取当前选中的槽位，如果没有选中则提示
        selected_items = self.position_list.selectedItems()
        if selected_items:
            slot_idx = self.position_list.row(selected_items[0])
        else:
            # 没有选中则保存第一个空位或第一个位置
            slot_idx = 0
            for i, pos in enumerate(self.saved_positions):
                if pos is None:
                    slot_idx = i
                    break

        # 保存当前位置和旋转
        translation = self.brace_transform.get_translation()
        rotation = self.brace_transform.get_rotation()
        self.saved_positions[slot_idx] = (translation.copy(), rotation.copy())
        self._refresh_position_list()
        self.position_list.setCurrentRow(slot_idx)

        # 保存到文件
        if self._current_brace_filepath:
            self.position_file_manager.save_positions(
                self._current_brace_filepath,
                self.saved_positions,
                self._inner_vertex_indices,
                self._inner_regions,
            )

        self.status_bar.showMessage(f"位置已保存到槽位 {slot_idx + 1} 并写入文件")

    def _load_position(self):
        """加载选中的位置"""
        selected_items = self.position_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要加载的位置")
            return

        slot_idx = self.position_list.row(selected_items[0])
        if self.saved_positions[slot_idx] is None:
            QMessageBox.information(self, "提示", "该槽位为空")
            return

        translation, rotation = self.saved_positions[slot_idx]

        # 应用位置
        self.brace_transform.reset()
        self.brace_transform.translate(*translation.tolist())
        if np.any(rotation):
            self.brace_transform.rotate_x(rotation[0])
            self.brace_transform.rotate_y(rotation[1])
            self.brace_transform.rotate_z(rotation[2])

        self._update_brace_view()
        self._render()
        self.status_bar.showMessage(f"已加载位置 {slot_idx + 1}")

        # 如果之前计算过距离，询问是否重新计算
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
        """清空选中的位置"""
        selected_items = self.position_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要清空的位置")
            return

        slot_idx = self.position_list.row(selected_items[0])
        self.saved_positions[slot_idx] = None
        self._refresh_position_list()
        self.status_bar.showMessage(f"位置 {slot_idx + 1} 已清空")

    def _on_position_double_clicked(self, item):
        """双击位置项 - 自动加载该位置"""
        slot_idx = self.position_list.row(item)
        if self.saved_positions[slot_idx] is None:
            self.status_bar.showMessage(f"位置 {slot_idx + 1} 为空")
            return

        translation, rotation = self.saved_positions[slot_idx]

        # 应用位置
        self.brace_transform.reset()
        self.brace_transform.translate(*translation.tolist())
        if np.any(rotation):
            self.brace_transform.rotate_x(rotation[0])
            self.brace_transform.rotate_y(rotation[1])
            self.brace_transform.rotate_z(rotation[2])

        self._update_brace_view()
        self._render()

        # 如果之前计算过距离，询问是否重新计算
        if self.current_distances is not None:
            reply = QMessageBox.question(
                self, "重新计算距离",
                "是否根据新位置重新计算距离？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._compute_distance()

    # ================================================================

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
        """切换测量模式"""
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
        """处理测量模式下的鼠标点击"""
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(display_x, display_y, 0, self.scene.renderer)

        actor = picker.GetActor()
        # 只接受足部或护具模型 Actor，忽略文字标签等 2D 元素
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
        """测量模式下鼠标移动，实时显示光标 3D 坐标"""
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

    def _undo_transform(self):
        if self.brace_transform.undo():
            self._update_brace_view()
            self._render()
            self.status_bar.showMessage("已撤销")

    def _screenshot(self):
        """保存截图"""
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
