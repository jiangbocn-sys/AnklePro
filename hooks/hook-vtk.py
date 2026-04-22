"""
PyInstaller hook for VTK
确保 VTK 模块被正确打包
"""
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_dynamic_libs,
    collect_data_files,
    get_module_file_attribute
)

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------- 初始化 PyInstaller 要求的全局变量 ----------
binaries = []   # 必须初始化，否则后面 .extend 会报 NameError
datas = []      # 也统一初始化

# ---------- 收集 vtk 的所有子模块 ----------
hiddenimports = collect_submodules('vtk')

# ---------- 收集数据文件 ----------
try:
    datas.extend(collect_data_files('vtk'))
except Exception:
    pass

# ---------- 手动添加特定隐藏导入 ----------
hiddenimports += [
    'vtk.qt',
    'vtk.qt.QVTKRenderWindowInteractor',
    'vtkRenderingCore',
    'vtkRenderingCore_python',
    'vtkInteractionStyle',
    'vtkInteractionWidgets',
    'vtkRenderingQt',
    'vtkGUISupportQt',
    'vtkCommonCore',
    'vtkCommonDataModel',
    'vtkFiltersCore',
    'vtkFiltersGeneral',
    'vtkImagingCore',
    'vtkImagingHybrid',
    'vtkIOCore',
    'vtkIOGeometry',
    'vtkIOPolygon',
    'vtkIOXML',
    'vtkCommonComputationalGeometry',
    'vtkFiltersExtraction',
    'vtkFiltersFlowPaths',
    'vtkFiltersGeometry',
    'vtkFiltersHypre',
    'vtkFiltersIMR',
    'vtkFiltersModeling',
    'vtkFiltersParallel',
    'vtkFiltersParallelDIY2',
    'vtkFiltersPoints',
    'vtkFiltersProgrammable',
    'vtkFiltersSelection',
    'vtkFiltersSMP',
    'vtkFiltersSources',
    'vtkFiltersStatistics',
    'vtkFiltersTexture',
    'vtkFiltersGeneralTF',
    'vtkFiltersVerdict',
    'vtkIOImage',
    'vtkIOImport',
    'vtkIOInfovis',
    'vtkIOLegacy',
    'vtkIOParallel',
    'vtkIOParallelXML',
    'vtkIOPLY',
    'vtkIOStreamH',
    'vtkIOTesting',
    'vtkIOVertexGraph',
    'vtkIOXdmf2',
    'vtkParallelCore',
    'vtkViewsCore',
    'vtkWrappingPythonCore',
]

# ---------- PyQt5 子模块 ----------
hiddenimports += collect_submodules('PyQt5')

# ---------- vtkmodules.qt 子模块（新版 VTK） ----------
hiddenimports += collect_submodules('vtkmodules.qt')

# ---------- 收集动态库（DLL/.pyd） ----------
try:
    binaries.extend(collect_dynamic_libs('vtk'))
except Exception:
    pass
try:
    binaries.extend(collect_dynamic_libs('vtkmodules'))
except Exception:
    pass
try:
    binaries.extend(collect_dynamic_libs('vtkmodules.qt'))
except Exception:
    pass

# ---------- 额外数据文件 ----------
try:
    datas.extend(collect_data_files('vtkmodules.qt'))
except Exception:
    pass

print(f"VTK hook: collected {len(hiddenimports)} hidden imports, {len(binaries)} binaries, {len(datas)} datas")