"""UI builder mixin — constructs the sidebar navigation and page stack"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtWidgets import QScrollArea
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from src.config import PRESET_THRESHOLDS


class UIBuilderMixin:
    """Mixin providing _setup_ui and navigation helpers for MainWindow."""

    def _setup_ui(self):
        """设置主窗口布局"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ============ 侧边导航栏（窗口最左侧）============
        sidebar = QWidget()
        sidebar.setFixedWidth(40)
        sidebar.setStyleSheet("background-color: #1a1a2e;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(4, 8, 4, 4)
        side_layout.setSpacing(4)
        side_layout.addSpacing(4)

        # 导航按钮 — 纵向文字（每个字一行）
        self._nav_buttons: list = []
        self._nav_page_to_button: dict = {}  # page index → button index
        nav_items = [
            ("模型加载", 0),
            ("内侧面", 1),
            ("贴合分析", 2),
            ("尺寸修改", 5),
            ("统计结果", 4),
            ("显示设置", 3),
        ]
        for text, idx in nav_items:
            btn = QPushButton("\n".join(text))
            btn.setCheckable(True)
            btn.setFixedWidth(32)
            btn.setMinimumHeight(44)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(self._nav_button_style(False))
            btn.clicked.connect(lambda _, i=idx: self._switch_page(i))
            side_layout.addWidget(btn)
            self._nav_buttons.append(btn)
            self._nav_page_to_button[idx] = len(self._nav_buttons) - 1
            if text in ("模型加载",):
                side_layout.addSpacing(8)
        side_layout.addStretch()

        self._nav_buttons[0].setChecked(True)
        self._nav_buttons[0].setStyleSheet(self._nav_button_style(True))

        main_layout.addWidget(sidebar)

        # --- 内容面板 + 3D视图：使用可调节分隔器 ---
        splitter = QSplitter(Qt.Horizontal)

        # 内容面板（左侧，堆叠页面 + 滚动容器）
        page_widget = QWidget()
        page_layout = QVBoxLayout(page_widget)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        self.stack = QStackedWidget()
        page_layout.addWidget(self.stack)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(page_widget)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        splitter.addWidget(scroll_area)

        # 3D视图（右侧）
        self.vtk_widget = QVTKRenderWindowInteractor(central)
        splitter.addWidget(self.vtk_widget)

        splitter.setSizes([300, 900])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        main_layout.addWidget(splitter, stretch=1)

        # ============ 页面 0: 模型加载 ============
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        model_layout.setContentsMargins(6, 6, 6, 6)

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

        # 加载按钮
        btn_load_foot = QPushButton("加载足部 STL")
        btn_load_foot.clicked.connect(self._load_foot)
        btn_load_brace = QPushButton("加载护具 STL")
        btn_load_brace.clicked.connect(self._load_brace)
        btn_load_step = QPushButton("加载护具 STEP 文件")
        btn_load_step.clicked.connect(self._load_brace_step)
        btn_convert_brace = QPushButton("将已加载护具转换为 STEP")
        btn_convert_brace.clicked.connect(self._convert_brace_to_step)

        load_btn_layout = QVBoxLayout()
        load_btn_layout.addWidget(btn_load_foot)
        load_btn_layout.addWidget(btn_load_brace)
        load_btn_layout.addWidget(btn_load_step)
        load_btn_layout.addWidget(btn_convert_brace)
        model_layout.addLayout(load_btn_layout)

        model_layout.addSpacing(12)

        # 清除空间（红色按钮）
        btn_clear = QPushButton("清除空间")
        btn_clear.clicked.connect(self._clear_workspace)
        btn_clear.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 8px;")
        model_layout.addWidget(btn_clear)

        model_layout.addStretch()
        self.stack.addWidget(model_tab)

        # ============ 页面 1: 内侧面 ============
        inner_page = QWidget()
        inner_page_layout = QVBoxLayout(inner_page)
        inner_page_layout.setContentsMargins(6, 6, 6, 6)

        # 选取按钮
        btn_pick_inner = QPushButton("选取内侧面")
        btn_pick_inner.clicked.connect(self._pick_inner_surface)
        btn_pick_inner.setMinimumHeight(40)
        inner_page_layout.addWidget(btn_pick_inner)

        # 内侧面统计信息
        self.lbl_inner_stats = QLabel("内侧面: 未选取")
        self.lbl_inner_stats.setWordWrap(True)
        self.lbl_inner_stats.setStyleSheet("color: #888; padding: 4px;")
        inner_page_layout.addWidget(self.lbl_inner_stats)

        # 区域列表
        inner_list_group = QGroupBox("内侧面区域")
        inner_list_layout = QVBoxLayout()
        self.region_list = QListWidget()
        self.region_list.setSelectionMode(QListWidget.NoSelection)
        self.region_list.itemChanged.connect(self._on_region_toggled)
        self.region_list.setMaximumHeight(200)
        inner_list_layout.addWidget(self.region_list)
        inner_list_group.setLayout(inner_list_layout)
        inner_page_layout.addWidget(inner_list_group)

        # 区域操作按钮
        region_btn_layout = QHBoxLayout()
        btn_select_all = QPushButton("全选")
        btn_select_all.clicked.connect(self._select_all_regions)
        btn_deselect_all = QPushButton("全不选")
        btn_deselect_all.clicked.connect(self._deselect_all_regions)
        region_btn_layout.addWidget(btn_select_all)
        region_btn_layout.addWidget(btn_deselect_all)
        inner_page_layout.addLayout(region_btn_layout)

        inner_page_layout.addSpacing(8)
        inner_page_layout.addWidget(QLabel("提示：勾选区域后前往「贴合分析」进行距离计算"))

        inner_page_layout.addStretch()
        self.stack.addWidget(inner_page)

        # ============ 页面 2: 贴合分析 ============
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        analysis_layout.setContentsMargins(6, 6, 6, 6)

        # 护具位置控制
        ctrl_group = QGroupBox("护具位置控制")
        ctrl_layout = QVBoxLayout()

        ctrl_layout.addWidget(QLabel("移动步长 (mm):"))
        self.spin_move = QDoubleSpinBox()
        self.spin_move.setRange(0.01, 10.0)
        self.spin_move.setSingleStep(0.1)
        self.spin_move.setValue(self.move_step)
        self.spin_move.valueChanged.connect(self._on_move_step_changed)
        ctrl_layout.addWidget(self.spin_move)

        ctrl_layout.addWidget(QLabel("旋转步长 (度):"))
        self.spin_rotate = QDoubleSpinBox()
        self.spin_rotate.setRange(0.01, 10.0)
        self.spin_rotate.setSingleStep(0.1)
        self.spin_rotate.setValue(self.rotate_step)
        self.spin_rotate.valueChanged.connect(self._on_rotate_step_changed)
        ctrl_layout.addWidget(self.spin_rotate)

        self.btn_fine = QPushButton("进入精细调节模式")
        self.btn_fine.setCheckable(True)
        self.btn_fine.clicked.connect(self._toggle_fine_mode)
        ctrl_layout.addWidget(self.btn_fine)

        btn_reset = QPushButton("重置变换")
        btn_reset.clicked.connect(self._reset_transform)
        ctrl_layout.addWidget(btn_reset)

        ctrl_group.setLayout(ctrl_layout)
        analysis_layout.addWidget(ctrl_group)

        # 计算操作
        calc_group = QGroupBox("距离计算")
        calc_layout = QVBoxLayout()
        btn_calc = QPushButton("计算距离")
        btn_calc.clicked.connect(self._compute_distance)
        btn_calc.setMinimumHeight(40)
        btn_calc.setStyleSheet("font-weight: bold; font-size: 13px;")
        calc_layout.addWidget(btn_calc)

        calc_btn_row = QHBoxLayout()
        btn_optimize = QPushButton("自动最优位置")
        btn_optimize.clicked.connect(self._optimize_position)
        btn_radial = QPushButton("计算径向距离")
        btn_radial.clicked.connect(self._compute_radial_distance)
        calc_btn_row.addWidget(btn_optimize)
        calc_btn_row.addWidget(btn_radial)
        calc_layout.addLayout(calc_btn_row)

        calc_group.setLayout(calc_layout)
        analysis_layout.addWidget(calc_group)

        # 足部显示控制
        foot_group = QGroupBox("足部模型显示")
        foot_layout = QHBoxLayout()
        self.btn_foot_visible = QPushButton("隐藏足部")
        self.btn_foot_visible.setCheckable(True)
        self.btn_foot_visible.setChecked(True)
        self.btn_foot_visible.clicked.connect(self._toggle_foot_visibility)
        self.btn_foot_transparent = QPushButton("半透明")
        self.btn_foot_transparent.setCheckable(True)
        self.btn_foot_transparent.clicked.connect(self._set_foot_transparent)
        foot_layout.addWidget(self.btn_foot_visible)
        foot_layout.addWidget(self.btn_foot_transparent)
        foot_group.setLayout(foot_layout)
        analysis_layout.addWidget(foot_group)

        # 保存位置
        save_group = QGroupBox("保存位置 (5 个槽位)")
        save_group_layout = QVBoxLayout()
        self.position_list = QListWidget()
        self.position_list.setMaximumHeight(120)
        self.position_list.itemDoubleClicked.connect(self._on_position_double_clicked)
        self._refresh_position_list()
        save_group_layout.addWidget(self.position_list)

        save_btn_layout = QHBoxLayout()
        btn_save = QPushButton("保存")
        btn_save.clicked.connect(self._save_position)
        btn_load = QPushButton("加载")
        btn_load.clicked.connect(self._load_position)
        btn_clear_pos = QPushButton("清空")
        btn_clear_pos.clicked.connect(self._clear_position)
        save_btn_layout.addWidget(btn_save)
        save_btn_layout.addWidget(btn_load)
        save_btn_layout.addWidget(btn_clear_pos)
        save_group_layout.addLayout(save_btn_layout)

        save_group.setLayout(save_group_layout)
        analysis_layout.addWidget(save_group)

        analysis_layout.addStretch()
        self.stack.addWidget(analysis_tab)

        # ============ 页面 3: 显示设置 ============
        display_tab = QWidget()
        display_layout = QVBoxLayout(display_tab)
        display_layout.setContentsMargins(6, 6, 6, 6)

        # 颜色阈值
        color_group = QGroupBox("颜色阈值")
        color_layout = QVBoxLayout()
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
        color_group.setLayout(color_layout)
        display_layout.addWidget(color_group)

        # 工具
        tool_group = QGroupBox("工具")
        tool_layout = QHBoxLayout()
        self.btn_wireframe = QPushButton("切换线框/实体")
        self.btn_wireframe.clicked.connect(self._toggle_wireframe)
        self.btn_measure = QPushButton("测量距离")
        self.btn_measure.setCheckable(True)
        self.btn_measure.clicked.connect(self._toggle_measurement)
        tool_layout.addWidget(self.btn_wireframe)
        tool_layout.addWidget(self.btn_measure)
        tool_group.setLayout(tool_layout)
        display_layout.addWidget(tool_group)

        display_layout.addStretch()
        self.stack.addWidget(display_tab)

        # ============ 页面 4: 统计结果 ============
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        stats_layout.setContentsMargins(6, 6, 6, 6)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Menlo", 10))
        stats_layout.addWidget(self.stats_text)
        self.stack.addWidget(stats_tab)

        # ============ 页面 5: 尺寸修改 ============
        deform_tab = QWidget()
        deform_layout = QVBoxLayout(deform_tab)
        deform_layout.setContentsMargins(6, 6, 6, 6)

        # 变形区域选择提示
        self.lbl_deform_region = QLabel("变形区域：请先选取内侧面并勾选区域")
        self.lbl_deform_region.setWordWrap(True)
        deform_layout.addWidget(self.lbl_deform_region)

        # 变形模式选择
        mode_group = QGroupBox("变形模式")
        mode_layout = QVBoxLayout()

        self.combo_deform_mode = QComboBox()
        self.combo_deform_mode.addItems([
            "法向变形（质心放射）",
            "法向变形（Z轴圆柱）",
            "方向拉伸（沿指定方向）",
            "径向缩放（绕中轴线）",
            "自适应（目标间隙）",
        ])
        self.combo_deform_mode.currentIndexChanged.connect(self._on_deform_mode_changed)
        mode_layout.addWidget(self.combo_deform_mode)
        mode_group.setLayout(mode_layout)
        deform_layout.addWidget(mode_group)

        # 方向选择（仅方向拉伸模式可见）
        self.dir_group = QGroupBox("拉伸方向")
        dir_layout = QVBoxLayout()
        dir_layout.addWidget(QLabel("沿选中点的表面法线方向变形"))
        self._dir_custom = QComboBox()
        self._dir_custom.addItem("表面法线（默认）")
        self._dir_custom.addItem("X+ (1, 0, 0)")
        self._dir_custom.addItem("X- (-1, 0, 0)")
        self._dir_custom.addItem("Y+ (0, 1, 0)")
        self._dir_custom.addItem("Y- (0, -1, 0)")
        self._dir_custom.addItem("Z+ (0, 0, 1)")
        self._dir_custom.addItem("Z- (0, 0, -1)")
        dir_layout.addWidget(self._dir_custom)
        self.dir_group.setLayout(dir_layout)
        self.dir_group.setVisible(False)
        deform_layout.addWidget(self.dir_group)

        # 径向轴选择（仅径向缩放模式可见）
        self.radial_group = QGroupBox("径向轴")
        radial_layout = QVBoxLayout()
        self._radial_axis = QComboBox()
        self._radial_axis.addItem("PCA 中轴线（默认）")
        self._radial_axis.addItem("X 轴 (1, 0, 0)")
        self._radial_axis.addItem("Y 轴 (0, 1, 0)")
        self._radial_axis.addItem("Z 轴 (0, 0, 1)")
        radial_layout.addWidget(self._radial_axis)
        radial_layout.addWidget(QLabel("径向轴决定缩放的方向，模型沿垂直于轴的平面内缩放"))
        self.radial_group.setLayout(radial_layout)
        self.radial_group.setVisible(False)
        deform_layout.addWidget(self.radial_group)

        # 变形方向提示
        self.lbl_deform_direction = QLabel("方向：用偏移量正负号控制（正=向外扩张，负=向内收缩）")
        self.lbl_deform_direction.setStyleSheet("color: #888;")
        deform_layout.addWidget(self.lbl_deform_direction)

        # 变形选点控制（仅方向拉伸模式可见）
        self.point_group = QGroupBox("变形选点")
        point_layout = QVBoxLayout()
        point_layout.addWidget(QLabel("按方向键 ↑↓←→ 移动选点位置"))
        point_btn_row = QHBoxLayout()
        self.btn_auto_select_point = QPushButton("自动选距足部最近点")
        self.btn_auto_select_point.clicked.connect(self._select_closest_to_foot)
        point_btn_row.addWidget(self.btn_auto_select_point)
        self.lbl_deform_point = QLabel("未选点")
        self.lbl_deform_point.setStyleSheet("color: #aaa;")
        point_btn_row.addWidget(self.lbl_deform_point)
        point_layout.addLayout(point_btn_row)
        self.point_group.setLayout(point_layout)
        self.point_group.setVisible(False)
        deform_layout.addWidget(self.point_group)

        # 偏移量
        self.lbl_deform_offset = QLabel("偏移量 (mm):  正数=向外扩张，负数=向内收缩")
        deform_layout.addWidget(self.lbl_deform_offset)
        self.spin_deform_offset = QDoubleSpinBox()
        self.spin_deform_offset.setRange(-20.0, 20.0)
        self.spin_deform_offset.setSingleStep(0.1)
        self.spin_deform_offset.setDecimals(2)
        self.spin_deform_offset.setValue(0.5)
        deform_layout.addWidget(self.spin_deform_offset)

        # 衰减半径
        self.lbl_deform_decay = QLabel("衰减半径 (mm, 0=整个选区均匀变形):")
        deform_layout.addWidget(self.lbl_deform_decay)
        self.spin_deform_decay = QDoubleSpinBox()
        self.spin_deform_decay.setRange(0.0, 100.0)
        self.spin_deform_decay.setSingleStep(1.0)
        self.spin_deform_decay.setValue(0.0)
        deform_layout.addWidget(self.spin_deform_decay)

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
        self.stack.addWidget(deform_tab)

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
        side_layout.addSpacing(12)
        side_layout.addWidget(coord_group)


        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def _switch_page(self, index: int):
        """切换到指定页面，同时更新导航按钮高亮"""
        self.stack.setCurrentIndex(index)
        btn_idx = self._nav_page_to_button.get(index, 0)
        for i, btn in enumerate(self._nav_buttons):
            checked = (i == btn_idx)
            btn.setChecked(checked)
            btn.setStyleSheet(self._nav_button_style(checked))

    def _nav_button_style(self, checked: bool) -> str:
        """返回导航按钮的样式（纵向文字，标签样式）"""
        if checked:
            return (
                "background-color: #e94560; color: white; "
                "border: none; border-radius: 6px; font-size: 11px; "
                "font-weight: bold; padding: 4px 2px;"
            )
        return (
            "background-color: transparent; color: #a0a0b0; "
            "border: none; border-radius: 6px; font-size: 11px; "
            "padding: 4px 2px;"
        )

    def _clear_region_lists(self):
        """清除内侧面区域列表"""
        self.region_list.clear()

    def _safe_remove_actor(self, actor):
        """安全地移除 VTK Actor（编译环境下防止 None 或已删除 actor 导致闪退）"""
        if actor is None:
            return
        try:
            self.scene.renderer.RemoveActor(actor)
        except Exception:
            pass

    def _render(self):
        """刷新渲染"""
        if self.render_window:
            self.render_window.Render()
