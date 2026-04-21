"""
PyInstaller hook for vtk.qt
确保 VTK 的 Qt 模块被正确打包
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# 收集 vtk.qt 的所有子模块
hiddenimports = collect_submodules('vtk.qt')

# 收集 vtk 的数据文件
datas = collect_data_files('vtk')

# 额外添加可能遗漏的模块
hiddenimports += [
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
]
