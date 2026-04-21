# AnklePro — 开发文档

> 足部-护具3D贴合度可视化分析工具
> 版本: v0.1.0 (规划)
> 创建日期: 2026-04-19

---

## 一、项目概述

### 1.1 项目目标

AnklePro 是一个交互式3D可视化工具，用于分析脚踝护具与足部模型的贴合度。核心功能是：

- 加载并渲染足部3D模型（STL/STEP格式）和护具3D模型（STL/STEP格式）
- 在3D场景中以不同颜色直观展示护具内表面与足部表面之间的空间距离分布
- 支持用户通过鼠标和键盘交互操作，精确定位护具到最佳佩戴位置
- 实时计算并可视化距离变化，辅助护具设计和佩戴验证

### 1.2 使用场景

1. **护具设计验证**: 设计师加载足部扫描模型和护具CAD模型，检查护具内腔与足部表面的间隙分布
2. **佩戴位置优化**: 通过交互调整护具位置，找到最佳佩戴姿态
3. **贴合度量化**: 获取精确的距离统计数据（最小/平均/中位数/分布）

### 1.3 现有基础

项目目录 `~/anklepro` 已包含：
- `data/` — 多组足部+护具STL/STEP模型文件（~7GB）
- `freecad/` — FreeCAD工程文件
- `stl_match_analysis.py` — 离线匹配分析脚本（已实现KD-tree、位置搜索、可视化）
- 历史分析结果和报告文档

本次开发是在现有分析能力基础上，构建**实时交互式GUI工具**。

---

## 二、技术选型

### 2.1 核心技术栈

| 组件 | 技术选型 | 版本要求 | 用途 |
|------|---------|---------|------|
| GUI框架 | **PyQt5** | >=5.15 | 主窗口、工具栏、侧边面板 |
| 3D渲染引擎 | **VTK** | >=9.3 | 3D场景渲染、交互控制、体素着色 |
| 数值计算 | **NumPy** | >=1.24 | 矩阵运算、几何变换 |
| 空间搜索 | **SciPy** | >=1.10 | cKDTree快速最近邻查询 |
| STEP解析 | **pythonocc-core** | >=7.7 | STEP文件导入，转换为三角网格 |
| STL解析 | **numpy-stl** | >=3.1 | STL文件解析（备用，VTK内置亦可） |

### 2.2 选型理由

**VTK 作为3D渲染引擎**（而非PyOpenGL或Three.js）：
- 内置完整的3D交互器（旋转、平移、缩放）
- 支持GPU加速的Mapper和Actor体系
- 原生支持STL文件加载
- 提供 `vtkScalarBarActor` 可直接实现颜色映射条
- 支持 `vtkPolyData` 的逐顶点颜色设置
- 成熟稳定，医学可视化领域广泛使用

**PyQt5 作为GUI框架**（而非Tkinter或wxPython）：
- `QVTKRenderWindowInteractor` 与VTK无缝集成
- 丰富的控件库（滑块、颜色选择器、表格等）
- 信号槽机制便于事件驱动编程
- 支持自定义样式和布局

**pythonocc-core**（基于OpenCASCADE）：
- 开源CAD内核，精确解析STEP格式
- 可将STEP B-Rep面转换为三角网格
- 比FreeCAD Python API更轻量，无需启动FreeCAD进程

### 2.3 依赖安装

```bash
pip install PyQt5 vtk numpy scipy pythonocc-core numpy-stl
```

> macOS 注意事项:
> - VTK 需要 X11 或原生 Cocoa 支持，建议 `pip install vtk` 使用预编译包
> - pythonocc-core 在 macOS 可能需要 conda: `conda install -c conda-forge pythonocc-core`
> - 推荐使用 Python 3.10+ 虚拟环境

---

## 三、系统架构

### 3.1 模块划分

```
anklepro/
├── main.py                    # 程序入口，启动GUI
├── config.py                  # 全局配置（颜色阈值、默认参数）
├── core/
│   ├── model_loader.py        # 模型加载器（STL/STEP -> vtkPolyData）
│   ├── transform_manager.py   # 变换管理器（平移、旋转、缩放矩阵）
│   ├── surface_picker.py      # 表面选取工具（点选/框选/法向筛选）
│   └── distance_calculator.py # 距离计算器（KD-tree + 逐顶点距离）
├── render/
│   ├── scene_manager.py       # VTK场景管理（坐标轴、灯光、相机）
│   ├── model_actors.py        # 模型渲染Actor（顶点着色、透明度）
│   ├── color_mapper.py        # 颜色映射器（距离 -> 颜色）
│   └── overlay_renderer.py    # 覆盖层渲染（信息文本、距离标尺）
├── ui/
│   ├── main_window.py         # 主窗口布局
│   ├── toolbar.py             # 工具栏（打开文件、计算、重置）
│   ├── control_panel.py       # 右侧控制面板
│   ├── color_threshold_editor.py  # 颜色阈值编辑面板
│   ├── stats_panel.py         # 统计信息面板
│   └── coordinate_display.py  # 坐标显示面板
├── data/                      # 测试模型数据目录（已有）
└── DEVELOPMENT.md             # 本文档
```

### 3.2 数据流

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  STL/STEP    │────>│   ModelLoader    │────>│  vtkPolyData    │
│  文件加载     │     │  (解析+网格化)    │     │  (顶点+三角面)   │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
┌──────────────┐     ┌──────────────────┐     ┌────────▼────────┐
│  表面选取     │<────│  SurfacePicker   │<────│  足部模型顶点    │
│  (用户交互)   │     │  (法向/区域筛选)  │     │  (构建KD-tree)  │
└──────┬───────┘     └──────────────────┘     └────────┬────────┘
       │                                               │
       │                                        ┌──────▼──────┐
       │                                        │  cKDTree    │
       │                                        │  (足部表面)  │
       │                                        └──────┬──────┘
       │                                               │
┌──────▼───────┐     ┌──────────────────┐     ┌────────▼────────┐
│  距离颜色映射 │<────│DistanceCalculator│<────│  护具内侧面顶点  │
│  (渲染着色)   │     │  (最近邻查询)     │     │  (选取后的子集)  │
└──────┬───────┘     └──────────────────┘     └─────────────────┘
       │
       ▼
┌──────────────────┐
│  VTK渲染管线      │
│  (vtkPolyDataMapper│
│   + vtkActor)     │
└──────────────────┘
```

### 3.3 状态机

```
[IDLE]          — 初始状态，等待用户加载模型
  │
  ▼ [加载足部模型]
[FOOT_LOADED]   — 足部模型已加载，等待加载护具
  │
  ▼ [加载护具模型]
[BOTH_LOADED]   — 两个模型均已加载，等待用户选取表面
  │
  ▼ [选取护具内侧面]
[SURFACE_PICKED]— 表面选取完成，等待用户放置护具
  │
  ▼ [用户拖拽/旋转护具到合适位置]
[POSITIONED]    — 护具位置已确认，等待启动计算
  │
  ▼ [点击"开始计算"]
[COMPUTING]     — 正在计算距离（可能耗时数秒）
  │
  ▼ [计算完成]
[RESULT_READY]  — 距离已计算，颜色已映射，显示结果
  │
  ▼ [用户精细调节护具]
[FINE_TUNING]   — 实时计算并更新颜色（增量更新）
```

---

## 四、核心功能详细设计

### 4.1 模型加载模块 (model_loader.py)

#### 4.1.1 功能

- 支持 STL（ASCII/Binary）格式加载
- 支持 STEP 格式加载（通过 pythonocc-core 转换为网格）
- 加载后自动计算模型包围盒和质心
- 自动缩放模型到合适的显示尺寸

#### 4.1.2 STL 加载

```python
def load_stl(filepath: str) -> vtkPolyData:
    """
    加载STL文件，返回vtkPolyData
    
    使用 vtkSTLReader 读取，自动检测ASCII/Binary格式。
    加载后执行：
    1. vtkCleanPolyData — 合并重复顶点
    2. vtkPolyDataNormals — 重新计算法线
    3. 计算包围盒和质心
    """
```

#### 4.1.3 STEP 加载（Phase 2）

```python
def load_step(filepath: str, mesh_deflection: float = 0.1) -> vtkPolyData:
    """
    加载STEP文件，转换为vtkPolyData
    
    流程：
    1. pythonocc-core 读取STEP文件 (STEPControl_Reader)
    2. 提取 TopoDS_Shape 的所有面
    3. BRepMesh_IncrementalMesh 对每个面进行三角剖分
       - mesh_deflection 控制网格精度（默认0.1mm）
    4. 遍历三角面，构建顶点数组和面索引
    5. 转换为 vtkPolyData
    6. 执行 CleanPolyData + Normals
    """
```

#### 4.1.4 模型数据结构

```python
@dataclass
class ModelData:
    """加载后的模型数据"""
    polydata: vtkPolyData        # VTK几何数据
    filepath: str                # 原始文件路径
    name: str                    # 模型名称（文件名）
    bounding_box: tuple          # (min_x, min_y, min_z, max_x, max_y, max_z)
    centroid: np.ndarray         # 质心坐标 [x, y, z]
    vertex_count: int            # 顶点数
    triangle_count: int          # 三角面数
    original_vertices: np.ndarray # 原始顶点数组 (N, 3)
    kd_tree: Optional[cKDTree]   # 空间索引树（足部模型需要）
```

### 4.2 表面选取模块 (surface_picker.py)

#### 4.2.1 选取方式

提供三种选取方式：

1. **法向筛选**（推荐用于护具内侧面）：
   - 用户指定一个参考方向（如"朝向足部中心"）
   - 选取所有法线方向与参考方向夹角小于阈值的三角面
   - 适用于护具空腔内侧面：法线朝向内侧的面

2. **区域框选**：
   - 在3D视图中用鼠标拖拽绘制矩形区域
   - 选取区域内的所有三角面
   - 适用于不规则曲面的粗略选取

3. **点选+扩展**：
   - 用户点击模型上的一个点
   - 自动扩展选取与该点连通且法向相近的连续曲面
   - 适用于选取"外侧面"、"内侧面"等连续区域

#### 4.2.2 核心算法

```python
def pick_by_normal(polydata: vtkPolyData,
                   reference_direction: np.ndarray,
                   angle_threshold: float = 60.0) -> np.ndarray:
    """
    按法向筛选表面
    
    参数:
        polydata: 模型几何数据
        reference_direction: 参考方向向量 [x, y, z]
        angle_threshold: 最大夹角（度）
    
    返回:
        被选中顶点的索引数组
    """
    # 1. 获取每个顶点的法线
    normals = extract_normals(polydata)
    
    # 2. 计算每个法线与参考方向的夹角
    cos_angles = np.dot(normals, reference_direction)
    
    # 3. 筛选夹角小于阈值的顶点
    mask = np.abs(cos_angles) > np.cos(np.radians(angle_threshold))
    
    return np.where(mask)[0]

def pick_by_region_growing(polydata: vtkPolyData,
                           seed_point: np.ndarray,
                           normal_threshold: float = 30.0,
                           max_distance: float = float('inf')) -> np.ndarray:
    """
    区域生长选取 — 从种子点扩展连续曲面
    
    参数:
        polydata: 模型几何数据
        seed_point: 种子点坐标
        normal_threshold: 相邻面法向最大夹角
        max_distance: 从种子点的最大测地距离
    
    返回:
        被选中顶点的索引数组
    """
    # 1. 找到离种子点最近的三角面作为种子面
    # 2. BFS遍历相邻三角面
    # 3. 如果相邻面法向夹角 < threshold，加入选取集
    # 4. 可选：限制最大测地距离
    pass
```

#### 4.2.3 手动修正机制

自动识别（法向法）后，必须允许用户修正选取范围：

**第一版（Phase 1-3）：点选添加/移除**
- 选取结果以半透明红色覆盖显示
- 用户切换到"修正模式"后：
  - 点击模型表面未被选中的面 → 将该面及其连通区域加入选取
  - 点击已被选中的面 → 将该面及其连通区域从选取中移除
- 使用区域生长算法确保选取始终是连续平面

**第二版（Phase 5+）：边界线拖拽**
- 在选取区域的边界上提取边界边（仅被一个选中三角面使用的边）
- 将边界边渲染为可交互的控制线（高亮、加粗）
- 用户拖拽控制线上的点，扩展或收缩选取范围
- 拖拽时实时预览选取范围的变化，松开后确认

**关键约束：** 无论哪种方式，选取结果必须是连续的曲面，不能产生零散的面片孤岛。

#### 4.2.4 选取结果可视化

- 选取的表面高亮显示（半透明红色覆盖层）
- 未选取的表面降低透明度至 30%
- 选取边界高亮显示（黄色边框线）
- 选取状态实时反馈，状态栏显示选中面数/顶点数

### 4.3 变换管理模块 (transform_manager.py)

#### 4.3.1 变换操作

```python
class TransformManager:
    """管理模型的4x4变换矩阵"""
    
    def __init__(self):
        self.matrix = np.eye(4)  # 当前变换矩阵
        self.center = np.zeros(3)  # 旋转中心（模型质心）
    
    def translate(self, dx: float, dy: float, dz: float):
        """平移变换"""
        t = np.eye(4)
        t[:3, 3] = [dx, dy, dz]
        self.matrix = t @ self.matrix
    
    def rotate_x(self, angle_deg: float):
        """绕X轴旋转"""
        r = rotation_x_matrix(angle_deg)
        # 绕质心旋转：先平移到原点，旋转，再平移回去
        self.matrix = translate(-self.center) @ r @ translate(self.center) @ self.matrix
    
    def rotate_y(self, angle_deg: float):
        """绕Y轴旋转"""
    
    def rotate_z(self, angle_deg: float):
        """绕Z轴旋转"""
    
    def scale(self, factor: float):
        """均匀缩放"""
    
    def apply(self, vertices: np.ndarray) -> np.ndarray:
        """对顶点数组应用当前变换"""
        homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
        transformed = (self.matrix @ homogeneous.T).T
        return transformed[:, :3]
    
    def reset(self):
        """重置为单位矩阵"""
        self.matrix = np.eye(4)
    
    def get_translation(self) -> np.ndarray:
        """获取当前平移量"""
        return self.matrix[:3, 3]
    
    def get_relative_position(self, other: 'TransformManager') -> np.ndarray:
        """获取相对另一个变换的位移"""
        return self.matrix[:3, 3] - other.matrix[:3, 3]
```

#### 4.3.2 交互映射

| 操作方式 | 动作 | 变换 |
|---------|------|------|
| 鼠标左键拖拽 | 旋转 | 绕视点中心旋转（VTK默认交互器） |
| 鼠标右键拖拽 | 平移 | X-Y平面平移 |
| 鼠标滚轮 | 缩放 | 沿视线方向缩放 |
| 方向键 ↑↓ | 平移 | Y轴 ±1mm |
| 方向键 ←→ | 平移 | X轴 ±1mm |
| Shift + 方向键 | 平移 | Z轴 ±1mm |
| Ctrl + 方向键 | 旋转 | 绕Z轴 ±1度 |
| Alt + 方向键 | 旋转 | 绕X轴 ±1度 |
| Shift + Alt + 方向键 | 旋转 | 绕Y轴 ±1度 |
| +/- 键 | 微调步长 | 增减移动/旋转步长 |

> **精细调节模式**: 进入 RESULT_READY 状态后，步长自动从 1mm 降为 0.1mm，旋转从 1度 降为 0.1度，支持亚毫米级精确调整。

### 4.4 距离计算模块 (distance_calculator.py)

#### 4.4.1 核心算法 — 有符号距离

距离计算采用**有符号距离**方案：
- **正值**：护具顶点在足部模型外部，表示间隙
- **负值**：护具顶点在足部模型内部，表示穿透（干涉/过紧）

判断点在网格内外的方法：使用 `vtkSelectEnclosedPoints`（VTK内置）或射线法。优先使用 VTK 内置方法，实现简单且经过充分测试。

```python
class DistanceCalculator:
    """计算护具内侧面到足部表面的有符号距离"""
    
    def __init__(self, foot_polydata: vtkPolyData):
        """
        初始化 — 构建足部表面的KD-tree + 内外判断器
        
        参数:
            foot_polydata: 足部模型完整几何数据
        """
        vertices = extract_vertices(foot_polydata)
        self.kd_tree = cKDTree(vertices)
        
        # 用于判断点在网格内外
        self.enclosed_checker = vtkSelectEnclosedPoints()
        self.enclosed_checker.SetSurfaceData(foot_polydata)
        self.enclosed_checker.Initialize()
    
    def compute_signed_distances(self, brace_vertices: np.ndarray) -> np.ndarray:
        """
        计算护具每个顶点到足部表面的有符号距离
        
        参数:
            brace_vertices: 护具内侧面的顶点 (M, 3)
        
        返回:
            有符号距离数组 (M,)，单位mm
            正值 = 间隙，负值 = 穿透深度
        """
        # 1. KD-tree查询最近距离（无符号）
        distances, _ = self.kd_tree.query(brace_vertices, k=1)
        
        # 2. 判断每个点在足部模型内部还是外部
        inside_flags = np.array([
            self.enclosed_checker.IsInsideVertex(vtk_points, i)
            for i in range(len(brace_vertices))
        ], dtype=bool)
        
        # 3. 内部点取负值
        signed_distances = np.where(inside_flags, -distances, distances)
        return signed_distances
    
    def compute_with_stats(self, brace_vertices: np.ndarray) -> dict:
        """
        计算距离并返回统计信息
        
        返回:
            {
                'distances': np.ndarray,    # 逐顶点有符号距离
                'min': float,               # 最小距离（最深处穿透，负值）
                'max': float,               # 最大距离（最大间隙）
                'mean': float,              # 平均距离
                'median': float,            # 中位数距离
                'std': float,               # 标准差
                'penetration_count': int,   # 穿透顶点数
                'penetration_ratio': float, # 穿透比例 (%)
                'percentile_5': float,      # 5%分位数
                'percentile_95': float,     # 95%分位数
                'histogram': tuple,         # (counts, bin_edges)
            }
        """
```

#### 4.4.2 第一阶段精度说明

第一阶段（MVP）使用"护具顶点到足部最近顶点"的距离。对于密集网格（足部50万顶点，三角面平均间距约0.3mm），点到点距离的最大误差约0.15mm，在工程上完全可接受。

**V2 扩展**（后续版本）：升级为"点到三角面"的精确距离计算。使用 VTK 的 `vtkOBBTree` 或自定义射线-三角面求交算法，将精度提高到 0.01mm 级别，但计算量约增加 3 倍。仅在第一阶段验证整体流程后再考虑。

#### 4.4.3 性能优化

- **降采样**: 对于超密集网格（>100万顶点），自动降采样至合理密度
- **批量查询**: cKDTree 原生支持批量最近邻查询，一次处理所有护具顶点
- **增量更新**: 精细调节时，采用节流策略，每 200ms 计算一次。如果卡顿，改为"按键松开后计算"。
- **预计算**: 足部KD-tree仅构建一次，护具移动时复用
- **优先流畅感**: 实时计算优先保证用户操作流畅，宁可牺牲一点即时性也不要卡顿

#### 4.4.4 预期性能

| 模型规模 | KD-tree构建 | 单次查询（全量） | 降采样查询 |
|---------|-----------|----------------|-----------|
| 足部 ~50万顶点 | ~0.3s | - | - |
| 护具内侧面 ~20万顶点 | - | ~0.8s | ~0.1s (降25%) |
| 护具内侧面 ~50万顶点 | - | ~2.0s | ~0.2s (降25%) |

> 注意：以上为理论估算，需在目标机器上运行实际基准测试验证。建议 Phase 1 完成后先用现有数据跑一次性能测试，根据实际数据调整降采样策略。

### 4.5 颜色映射模块 (color_mapper.py)

#### 4.5.1 默认颜色方案（有符号距离）

```python
# config.py 中的默认配置
DEFAULT_DISTANCE_THRESHOLDS = [
    (-float('inf'), -2.0, (0.6, 0.0, 0.0), '穿透 (干涉)'),     # 深红: 穿透 > 2mm
    (-2.0,  0.0, (1.0, 0.2, 0.2), '穿透 (轻微)'),              # 亮红: 轻微穿透
    ( 0.0,  4.0, (1.0, 0.0, 0.0), '偏紧 (0-4mm)'),             # 红色: 间隙 < 4mm
    ( 4.0,  6.0, (0.0, 0.85, 0.0), '理想 (4-6mm)'),           # 绿色: 理想范围
    ( 6.0,  7.0, (1.0, 1.0, 0.0), '偏松 (6-7mm)'),             # 黄色: 开始偏松
    ( 7.0,  float('inf'), (1.0, 0.6, 0.0), '过松 (>7mm)'),     # 橙色: 过松
]
```

**工程依据说明：**
- 4mm 对应软垫厚度，理想状态是加软垫后护具对足部无向内压力
- 4-6mm 绿色区间：加4mm软垫后，护具与足部仍有 0-2mm 余量，适应人体不同时段足部状态的微小变化，佩戴舒适
- <4mm 红色：加软垫后空间不足，可能产生向内压力
- <0mm 深红色：护具已嵌入足部，即使不加软垫也会压迫
- >7mm 橙色：间隙过大，护具可能松动

#### 4.5.2 预设方案

UI 中提供预设方案下拉菜单，包含：

| 预设名称 | 绿色区间 | 适用场景 |
|---------|---------|---------|
| **默认（4mm软垫）** | 4-6mm | 标准软垫厚度 |
| **厚软垫（6mm）** | 6-8mm | 加厚软垫或冬季使用 |
| **薄软垫（2mm）** | 2-4mm | 薄软垫或夏季使用 |
| **自定义** | 用户设定 | 用户自行调整 |

切换预设方案时，自动更新颜色阈值和颜色映射。用户也可以基于任意预设进行微调。

#### 4.5.3 映射逻辑

```python
class ColorMapper:
    """将有符号距离值映射为RGB颜色"""
    
    def __init__(self, thresholds: list = None):
        """
        参数:
            thresholds: 有符号颜色阈值列表
                [(min_dist, max_dist, (r, g, b), label), ...]
                按 min_dist 升序排列，允许负值
        """
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.smooth_range = 0.5  # 阈值过渡范围 (mm)，用户可调
    
    def map_distance(self, distance: float) -> tuple:
        """单个距离值映射为 (R, G, B)"""
        for min_d, max_d, color, label in self.thresholds:
            if min_d < distance <= max_d:
                return color
        return self.thresholds[-1][2]
    
    def map_distances(self, distances: np.ndarray) -> np.ndarray:
        """
        批量映射 — 返回 (N, 3) 的RGB数组
        使用 NumPy 向量化操作，比逐元素映射快100倍+
        """
        colors = np.zeros((len(distances), 3))
        prev_min = -float('inf')
        for min_d, max_d, (r, g, b), _ in self.thresholds:
            mask = (distances > prev_min) & (distances <= max_d)
            colors[mask] = [r, g, b]
            prev_min = min_d
        return colors
    
    def set_thresholds(self, thresholds: list):
        """更新阈值配置（用户通过UI修改）"""
        self.thresholds = sorted(thresholds, key=lambda x: x[0])
    
    def set_smooth_range(self, range_mm: float):
        """设置平滑过渡范围 (0.1mm ~ 2.0mm)"""
        self.smooth_range = max(0.1, min(2.0, range_mm))
```

#### 4.5.4 平滑过渡

在阈值边界处，默认使用 0.5mm 的平滑过渡范围（线性插值），避免颜色突变。例如在 4mm 阈值处，3.75mm~4.25mm 范围内颜色从红色渐变到绿色。

- 过渡范围可通过 UI 滑块调整（0.1mm ~ 2.0mm）
- 用户也可切换为"硬阈值"模式（无过渡）
- 第一版默认开启平滑过渡

#### 4.5.5 颜色阈值编辑器 UI

右侧面板提供：
- **阈值表格**：可编辑每个范围的下限、上限值和颜色
- **颜色选择器**：点击颜色单元格弹出标准颜色选择对话框
- **添加/删除阈值范围按钮**：支持自定义更多颜色区间
- **预设方案下拉菜单**：快速切换预设（默认/厚软垫/薄软垫）
- **平滑过渡滑块**：0.1mm ~ 2.0mm 可调
- **实时预览色条**：显示当前配置的颜色分布效果
- **标签列**：每个范围显示文字说明（穿透/偏紧/理想/偏松/过松）

### 4.6 VTK 场景管理 (scene_manager.py)

#### 4.6.1 场景组件

```python
class SceneManager:
    """VTK渲染场景管理"""
    
    def __init__(self, render_window: vtkRenderWindow):
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.15)  # 深灰背景
        render_window.AddRenderer(self.renderer)
        
        # 坐标轴
        self.axes_actor = vtkAxesActor()
        self.axes_actor.SetTotalLength(50, 50, 50)  # 50mm轴长
        self.renderer.AddActor(self.axes_actor)
        
        # 地面网格（可选）
        self.grid_actor = None
        
        # 标量条（颜色图例）
        self.scalar_bar = vtkScalarBarActor()
        self.scalar_bar.SetTitle("距离 (mm)")
        self.renderer.AddActor2D(self.scalar_bar)
    
    def add_model(self, polydata: vtkPolyData, color: tuple = (0.8, 0.8, 0.8)) -> vtkActor:
        """添加模型到场景"""
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(0.7)
        
        self.renderer.AddActor(actor)
        return actor
    
    def update_model_colors(self, actor: vtkActor, colors: np.ndarray):
        """
        更新模型顶点颜色
        
        参数:
            actor: 要更新的模型Actor
            colors: (N, 3) RGB数组，与顶点数一致
        """
        polydata = actor.GetMapper().GetInput()
        
        # 创建颜色数组
        color_array = vtkUnsignedCharArray()
        color_array.SetName("DistanceColors")
        color_array.SetNumberOfComponents(3)
        
        for r, g, b in (colors * 255).astype(np.uint8):
            color_array.InsertNextTuple3(r, g, b)
        
        polydata.GetPointData().SetScalars(color_array)
        
        # 更新Mapper使用顶点颜色
        mapper = actor.GetMapper()
        mapper.SetScalarModeToUsePointData()
        mapper.Update()
        
        actor.GetProperty().SetOpacity(0.85)
    
    def add_text_overlay(self, text: str, position: tuple = (10, 10)):
        """添加2D文字覆盖层（坐标信息等）"""
        text_actor = vtkTextActor()
        text_actor.SetInput(text)
        text_actor.SetPosition(position)
        text_actor.GetTextProperty().SetFontSize(14)
        text_actor.GetTextProperty().SetColor(1, 1, 1)
        self.renderer.AddActor2D(text_actor)
        return text_actor
    
    def reset_camera(self):
        """重置相机到默认视角"""
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Elevation(30)
        self.renderer.GetActiveCamera().Azimuth(45)
```

### 4.7 坐标显示

在屏幕固定位置（建议左下角）显示：

```
┌─────────────────────────────────┐
│ 护具相对足部位置:                 │
│   ΔX = +12.3 mm                 │
│   ΔY = -5.7 mm                  │
│   ΔZ = +23.1 mm                 │
│                                 │
│ 护具旋转角度:                     │
│   Rx = 0.0°  Ry = 0.0°  Rz = 0.0°│
│                                 │
│ 步长: 移动 0.1mm | 旋转 0.1°     │
└─────────────────────────────────┘
```

每次护具位置变化时实时更新。

---

## 五、UI布局设计

### 5.1 主窗口布局

```
┌──────────────────────────────────────────────────────────────────┐
│  File    Edit    View    Tools    Help                           │ 菜单栏
├──────────────────────────────────────────────────────────────────┤
│  [📂打开] [📐重置视角] [🎯选取表面] [▶计算] [↩重置] [💾导出]       │ 工具栏
├──────────────────────────────────────┬───────────────────────────┤
│                                      │                           │
│                                      │  📋 模型列表               │
│                                      │  ☑ 足部模型 (foot.stl)     │
│                                      │  ☑ 护具模型 (exol.stl)     │
│                                      │                           │
│                                      │  🎨 颜色阈值               │
│                                      │  ┌─────┬──────┬─────────┐   │
│                                      │  │ 下限 │ 上限  │ 颜色     │   │
│                                      │  │ -∞  │ -2.0 │ ■ 深红   │   │
│                                      │  │ -2  │  0.0 │ ■ 亮红   │   │
│                                      │  │  0  │  4.0 │ ■ 红色   │   │
│                                      │  │  4  │  6.0 │ ■ 绿色   │   │
│                                      │  │  6  │  7.0 │ ■ 黄色   │   │
│                                      │  │  7  │  ∞   │ ■ 橙色   │   │
│                                      │  └─────┴──────┴─────────┘   │
│                                      │  [+ 添加] [- 删除]          │
│                                      │  预设: [▼ 默认4mm软垫]      │
│                                      │                           │
│                                      │  📊 统计信息               │
│                                      │  最小距离: 1.2 mm          │
│                                      │  平均距离: 4.8 mm          │
│                                      │  中位数:  4.3 mm           │
│                                      │  标准差:  1.7 mm           │
│                                      │                           │
│         3D 渲染视图                   │  📍 坐标信息               │
│      (VTK Render Window)             │  ΔX = +12.3 mm            │
│                                      │  ΔY = -5.7 mm             │
│    [坐标轴] [足部] [护具]              │  ΔZ = +23.1 mm            │
│    [颜色映射覆盖层]                   │                           │
│                                      │  ⚙ 操作模式               │
│                                      │  ◉ 整体移动                │
│                                      │  ○ 精细调节                │
│                                      │  步长: [0.1] mm           │
│                                      │                           │
│                                      │  📤 导出                   │
│                                      │  [保存截图] [导出报告]     │
│                                      │                           │
├──────────────────────────────────────┴───────────────────────────┤
│  状态栏: 就绪 | 足部: 482K顶点 | 护具: 523K顶点 | 选中内侧面: 186K顶点 │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl+O | 打开文件对话框 |
| Ctrl+R | 重置相机视角 |
| Ctrl+D | 计算距离 |
| Ctrl+S | 保存截图 |
| Ctrl+Z | 撤销上一步变换 |
| Ctrl+Y | 重做 |
| 1 | 选取足部表面 |
| 2 | 选取护具表面 |
| F | 切换线框/实体模式 |
| G | 显示/隐藏地面网格 |
| A | 显示/隐藏坐标轴 |
| Space | 开始/停止实时计算 |
| Esc | 取消当前操作 |

---

## 六、开发步骤与里程碑

> **总体策略**：先做最小可运行原型（MVP），验证技术路线后再逐步完善。Phase 1 仅支持 STL 文件，STEP 支持延后。

### Phase 1: MVP 基础框架（预计 3-5 天）

- [ ] **1.1** 在 `~/anklepro/src/` 创建项目目录结构
- [ ] **1.2** 配置 Python 虚拟环境和依赖安装（venv + pip）
- [ ] **1.3** 实现 `main.py` 入口和 PyQt5 主窗口（空框架）
- [ ] **1.4** 嵌入 VTK `QVTKRenderWindowInteractor`
- [ ] **1.5** 实现坐标轴显示和基本3D交互（VTK默认交互器）
- [ ] **1.6** 实现 `model_loader.py` — STL文件加载（Binary + ASCII 自动检测）
- [ ] **1.7** 实现文件打开对话框，支持选择足部模型和护具模型
- [ ] **1.8** 实现基础距离计算：加载两个模型后，自动计算护具全部顶点到足部的距离并着色

**验收标准**: 能打开两个STL模型，在3D视图中显示，鼠标可旋转/平移/缩放，点击计算后护具按距离着色

### Phase 2: 模型变换与STEP支持（预计 3-4 天）

- [ ] **2.1** 实现 `transform_manager.py` — 4x4变换矩阵
- [ ] **2.2** 实现护具模型的独立变换（足部模型固定不动）
- [ ] **2.3** 键盘事件映射（方向键平移，组合键旋转）
- [ ] **2.4** 坐标显示面板实时更新（护具相对足部中心点的偏移）
- [ ] **2.5** 撤销/重做功能（变换历史栈）
- [ ] **2.6** STEP 文件加载（pythonocc-core 转换三角网格）

**验收标准**: 能用键盘精确移动和旋转护具，坐标信息实时更新；能加载STEP格式护具模型

### Phase 3: 表面选取（预计 3-4 天）

- [ ] **3.1** 实现法向筛选算法 — 自动识别护具内侧面（法线朝向质心）
- [ ] **3.2** 实现区域框选（VTK widget）
- [ ] **3.3** 实现点选+区域生长算法
- [ ] **3.4** 选取结果可视化（高亮/透明度变化/边界高亮）
- [ ] **3.5** 选取修正模式 — 点选添加/移除面（第一版，边界拖拽延后）
- [ ] **3.6** 距离计算改为仅对选取的表面进行

**验收标准**: 能自动识别护具内侧面，支持用户手动修正，距离计算仅针对选取表面

### Phase 4: 颜色映射与统计（预计 3-4 天）

- [ ] **4.1** 实现有符号距离计算（正值=间隙，负值=穿透）
- [ ] **4.2** 实现 `color_mapper.py` — 有符号距离到颜色的映射
- [ ] **4.3** 实现 VTK 顶点颜色更新管线（支持负值着色）
- [ ] **4.4** 实现标量条（ScalarBarActor）显示，标注正负值
- [ ] **4.5** 实现颜色阈值编辑器 UI（可编辑范围、颜色、标签）
- [ ] **4.6** 实现预设方案切换（默认/厚软垫/薄软垫/自定义）
- [ ] **4.7** 实现平滑过渡（用户可调范围 0.1-2.0mm）
- [ ] **4.8** 实现统计信息面板（含穿透计数/比例）

**验收标准**: 护具内侧面按有符号距离正确着色（含穿透深红色），阈值可编辑，预设可切换

### Phase 5: 实时交互与优化（预计 3-4 天）

- [ ] **5.1** 实现精细调节模式（步长自动降低：移动0.1mm，旋转0.1度）
- [ ] **5.2** 实现实时距离更新 — 移动护具时每200ms自动计算
- [ ] **5.3** 性能优化 — 降采样 + 批量查询 + 节流
- [ ] **5.4** 添加计算进度指示器
- [ ] **5.5** 实现导出截图和距离报告功能
- [ ] **5.6** 选取边界线拖拽（第二版交互，Phase 3点选的升级）

**验收标准**: 移动护具时颜色实时变化，无卡顿；支持边界拖拽修正选取范围

### Phase 6: 完善与优化（预计 2-3 天）

- [ ] **6.1** 添加键盘快捷键（完整的快捷键表）
- [ ] **6.2** 完善状态栏和错误提示
- [ ] **6.3** 内存优化（大模型降采样显示）
- [ ] **6.4** 运行实际性能基准测试，根据数据调优
- [ ] **6.5** 编写用户使用文档
- [ ] **6.6** 打包和分发准备

**验收标准**: 完整可用，用户体验流畅，性能达标

---

## 七、关键技术难点与解决方案

### 7.1 STEP文件三角剖分精度（Phase 2）

**问题**: STEP 是精确B-Rep格式，转换为三角网格时会损失精度。

**方案**:
- 使用 `BRepMesh_IncrementalMesh`，设置 `deflection=0.05mm`（约为足部扫描精度的一半）
- 对于超大模型，先检测网格面数，超过阈值时自动调整 deflection
- 缓存三角剖分结果，避免重复计算

### 7.2 护具内侧面的自动识别与手动修正

**问题**: 如何自动识别护具的"空腔内侧面"？对于非标准空腔结构，自动识别可能不准确。

**方案**:
- **自动识别（法向法）**: 计算护具质心，选取所有法线朝向质心的三角面作为初始选取
- **用户引导（区域生长）**: 用户点击内侧面一个点，自动扩展选取连通且法向相近的连续曲面
- **手动修正（点选）**: 自动选取后，用户可点击添加/移除面，确保选取的是正确的连续平面
- **推荐**: 法向法作为默认，自动识别后立即高亮显示选取结果，允许用户修正

```python
def identify_inner_surface(polydata: vtkPolyData) -> np.ndarray:
    """识别护具空腔内侧面"""
    centroid = compute_centroid(polydata)
    normals = extract_normals(polydata)
    centers = extract_triangle_centers(polydata)
    
    # 从面中心指向质心的向量
    to_centroid = centroid - centers
    
    # 法线与指向质心方向夹角 < 90度 的为内侧面
    dot_products = np.sum(normals * to_centroid, axis=1)
    inner_mask = dot_products > 0
    
    return np.where(inner_mask)[0]
```

### 7.3 实时计算性能

**问题**: 护具移动时重新计算距离可能耗时1-2秒，影响实时性。有符号距离计算（含内外判断）额外增加约30%耗时。

**方案**:
- **降采样**: 将护具内侧面顶点降采样至 5-10 万点（不影响视觉精度）
- **KD-tree复用**: 足部KD-tree只构建一次
- **vtkSelectEnclosedPoints 优化**: 该判断器初始化后，批量查询效率较高，但仍建议在后台线程执行
- **异步计算**: 在后台线程计算，不阻塞UI
- **节流**: 用户连续操作时，每200ms计算一次，而非每次按键都计算
- **优先流畅感**: 如果200ms节流仍卡顿，改为"按键松开后计算"
- **增量更新**: 检测位移超过阈值（如0.5mm）时才触发重算

### 7.4 超大模型内存管理

**问题**: 数据目录中的模型文件最大达686MB（STEP），加载后内存占用可能超过4GB。

**方案**:
- 加载时自动检测顶点数，超过100万顶点时自动降采样
- STEP加载时使用自适应 deflection，控制网格密度
- 使用 `vtkCleanPolyData` 合并重复顶点，减少内存
- 提供"降低显示精度"选项，牺牲渲染质量换取流畅度

### 7.5 颜色过渡平滑化

**问题**: 简单的阈值分段颜色在边界处会产生突变。

**方案**:
- 默认模式：使用平滑插值（在阈值附近 0.5mm 范围内线性过渡）
- 高级模式：提供连续颜色映射（如 `matplotlib` 的 colormap）
- 可选开关：用户可选择"硬阈值"或"平滑过渡"模式

```python
def map_distance_smooth(distance: float, thresholds: list) -> tuple:
    """带平滑过渡的颜色映射"""
    # 在每个阈值边界附近 0.5mm 范围内进行线性插值
    SMOOTH_RANGE = 0.5  # mm
    ...
```

---

## 八、配置文件格式

### 8.1 默认配置 (config.py)

```python
# config.py

# 颜色阈值配置 — 有符号距离（用户可在UI中修改）
# 格式: (min_dist, max_dist, (r, g, b), label)
DEFAULT_DISTANCE_THRESHOLDS = [
    (-float('inf'), -2.0, (0.6, 0.0, 0.0), '穿透 (干涉)'),
    (-2.0,  0.0, (1.0, 0.2, 0.2), '穿透 (轻微)'),
    ( 0.0,  4.0, (1.0, 0.0, 0.0), '偏紧'),
    ( 4.0,  6.0, (0.0, 0.85, 0.0), '理想'),
    ( 6.0,  7.0, (1.0, 1.0, 0.0), '偏松'),
    ( 7.0,  float('inf'), (1.0, 0.6, 0.0), '过松'),
]

# 预设方案
PRESET_SCHEMES = {
    'default': {
        'name': '默认（4mm软垫）',
        'ideal_range': (4.0, 6.0),
    },
    'thick': {
        'name': '厚软垫（6mm）',
        'ideal_range': (6.0, 8.0),
    },
    'thin': {
        'name': '薄软垫（2mm）',
        'ideal_range': (2.0, 4.0),
    },
}

# 交互设置
DEFAULT_MOVE_STEP = 1.0        # 默认移动步长 (mm)
FINE_MOVE_STEP = 0.1           # 精细移动步长 (mm)
DEFAULT_ROTATE_STEP = 1.0      # 默认旋转步长 (度)
FINE_ROTATE_STEP = 0.1         # 精细旋转步长 (度)

# 渲染设置
BACKGROUND_COLOR = (0.1, 0.1, 0.15)  # 深灰蓝
MODEL_OPACITY_NORMAL = 0.7     # 正常模式透明度
MODEL_OPACITY_RESULT = 0.85    # 结果模式透明度
AXES_LENGTH = 50.0             # 坐标轴长度 (mm)

# 性能设置
MAX_VERTICES_DISPLAY = 500_000  # 最大显示顶点数
KD_TREE_DOWNSAMPLE = 8         # KD-tree构建时的降采样因子
REALTIME_THROTTLE_MS = 200     # 实时计算节流间隔 (ms)
REALTIME_MOVE_THRESHOLD = 0.5  # 触发实时计算的最小位移 (mm)

# 平滑过渡设置
SMOOTH_RANGE_DEFAULT = 0.5     # 阈值过渡范围 (mm)
SMOOTH_RANGE_MIN = 0.1         # 最小过渡范围
SMOOTH_RANGE_MAX = 2.0         # 最大过渡范围

# STEP网格设置（Phase 2启用）
STEP_MESH_DEVIATION = 0.1      # STEP三角剖分偏差 (mm)
```

---

## 九、导出功能

### 9.1 截图导出

```python
def export_screenshot(render_window: vtkRenderWindow, filepath: str):
    """导出当前视图截图"""
    window_to_image = vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetScale(2)  # 2x分辨率
    
    writer = vtkPNGWriter()
    writer.SetFileName(filepath)
    writer.SetInputConnection(window_to_image.GetOutputPort())
    writer.Write()
```

### 9.2 距离报告导出

```python
def export_distance_report(stats: dict, config: dict, filepath: str):
    """导出距离分析报告（Markdown格式）"""
    report = f"""# 护具贴合度分析报告

## 模型信息
- 足部模型: {config['foot_file']}
- 护具模型: {config['brace_file']}

## 护具位置
- ΔX = {config['dx']:.2f} mm
- ΔY = {config['dy']:.2f} mm
- ΔZ = {config['dz']:.2f} mm

## 有符号距离统计
| 指标 | 值 |
|------|-----|
| 最小距离 | {stats['min']:.2f} mm |
| 最大距离 | {stats['max']:.2f} mm |
| 平均距离 | {stats['mean']:.2f} mm |
| 中位数 | {stats['median']:.2f} mm |
| 标准差 | {stats['std']:.2f} mm |
| 穿透顶点数 | {stats['penetration_count']} |
| 穿透比例 | {stats['penetration_ratio']:.1f}% |

## 颜色阈值方案
{format_thresholds(config['thresholds'])}

## 各区间占比
| 区间 | 占比 | 颜色 |
|------|------|------|
| 穿透 | {stats.get('penetration_ratio', 0):.1f}% | 深红/亮红 |
| 偏紧 (0-4mm) | {stats.get('tight_ratio', 0):.1f}% | 红色 |
| 理想 (4-6mm) | {stats.get('ideal_ratio', 0):.1f}% | 绿色 |
| 偏松 (6-7mm) | {stats.get('loose_ratio', 0):.1f}% | 黄色 |
| 过松 (>7mm) | {stats.get('very_loose_ratio', 0):.1f}% | 橙色 |
"""
    with open(filepath, 'w') as f:
        f.write(report)
```

---

## 十、测试计划

### 10.1 单元测试

| 测试模块 | 测试内容 |
|---------|---------|
| model_loader | STL/STEP加载正确性，顶点数验证 |
| transform_manager | 矩阵运算正确性，复合变换验证 |
| distance_calculator | KD-tree查询精度，与暴力法对比 |
| color_mapper | 阈值映射正确性，边界值测试 |
| surface_picker | 法向筛选准确性，区域生长连通性 |

### 10.2 集成测试

| 测试场景 | 预期结果 |
|---------|---------|
| 加载 tangle_LEFT_foot.stl + tangle_LEFT_exol.stl | 两个模型正确显示 |
| 选取护具内侧面后计算 | 内侧面正确着色 |
| 移动护具后重新计算 | 颜色实时更新 |
| 修改颜色阈值 | 颜色映射立即更新 |
| STEP文件加载 | 与STL加载结果一致 |

### 10.3 性能测试

| 测试项目 | 目标 |
|---------|------|
| 50万顶点模型加载时间 | < 3秒 |
| 首次距离计算时间（20万护具顶点） | < 2秒 |
| 实时计算帧率 | > 5 FPS |
| 内存占用峰值 | < 4GB |

---

## 十一、已知数据资源

项目 `data/` 目录已有模型可用于开发测试：

| 文件 | 类型 | 大小 | 用途 |
|------|------|------|------|
| `tangle_LEFT_foot.stl` | 足部 STL | 5.2MB | 测试用足部模型 |
| `tangle_LEFT_exol.stl` | 护具 STL | 28MB | 测试用护具模型 |
| `tangle_LEFT_exol.step` | 护具 STEP | 655MB | 测试STEP加载 |
| `jiangbo_Left_foot.stl` | 足部 STL | 4.7MB | 波哥个人足部数据 |
| `jiangbo_LEFT_exol.stl` | 护具 STL | 28MB | 波哥个人护具数据 |
| `jiangbo_LEFT_exol.step` | 护具 STEP | 655MB | 波哥护具STEP格式 |

> 注意: 部分 `.stl` 文件实际是 ASCII 格式，部分为 Binary 格式，加载器需自动检测。

---

## 十二、后续扩展方向

1. **多护具对比**: 同时加载多个护具模型，对比贴合效果
2. **软垫模拟**: 在距离计算中考虑软垫厚度，模拟加垫后的贴合情况
3. **自动最佳位置搜索**: 基于现有 `stl_match_analysis.py` 的搜索算法，集成自动定位
4. **3D打印导出**: 根据距离分布，自动生成护具厚度调整建议
5. **批量分析**: 对多组足部-护具数据进行批量分析
6. **Web版本**: 使用 Three.js / Pyodide 实现浏览器端访问

---

*文档版本: v1.0 | 2026-04-19 | AnklePro 项目开发规划*
