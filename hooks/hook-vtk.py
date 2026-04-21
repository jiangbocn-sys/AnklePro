"""
PyInstaller hook for VTK
确保 VTK 模块被正确打包
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, get_module_file_attribute

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 收集 vtk 的所有子模块
hiddenimports = collect_submodules('vtk')

# 收集 vtk 的数据文件
datas = []

try:
    datas = collect_data_files('vtk')
except Exception:
    pass

# 确保包含 vtk.qt 模块
hiddenimports += [
    'vtk.qt',
    'vtk.qt.QVTKRenderWindowInteractor',
]

# VTK 渲染模块
hiddenimports += [
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

# 收集 PyQt5 相关
hiddenimports += collect_submodules('PyQt5')

print(f"VTK hook: 收集了 {len(hiddenimports)} 个隐藏导入")
