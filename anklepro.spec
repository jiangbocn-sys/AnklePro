# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置
使用方法:
    pyinstaller anklepro.spec

输出:
    dist/AnklePro.exe - Windows 可执行文件
"""

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# ------------------------------------------------------------
# 1. 初始化要传递给 Analysis 的列表
# ------------------------------------------------------------
# 原有二进制文件（目前为空）
binaries = []

# 原有数据文件（你手动指定的）
datas = [
    ('src/config.py', 'src'),
    ('src/core', 'src/core'),
    ('src/render', 'src/render'),
    ('src/ui', 'src/ui'),
]

# 原有隐藏导入（你手动指定的）
hiddenimports = [
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'vtk',
    'vtk.qt',
    'vtk.qt.QVTKRenderWindowInteractor',
    'vtkRenderingCore',
    'vtkRenderingCore_python',
    'vtkInteractionStyle',
    'vtkInteractionWidgets',
    'vtkRenderingQt',
    'vtkGUISupportQt',
    'numpy',
    'numpy.core',
    'numpy.core._methods',
    'numpy.core._dtype_ctypes',
    'numpy.random',
    'scipy',
    'scipy.spatial',
    'scipy.spatial.ckdtree',
    'scipy._lib.messagestream',
    'scipy._lib',
]

# ------------------------------------------------------------
# 2. 使用 collect_all 强制收集 VTK 和 PyQt5 的所有依赖
# ------------------------------------------------------------
vtk_binaries, vtk_datas, vtk_hiddenimports = collect_all('vtkmodules')
pyqt_binaries, pyqt_datas, pyqt_hiddenimports = collect_all('PyQt5')

# 将 collect_all 的结果合并到原有列表中
binaries.extend(vtk_binaries)
binaries.extend(pyqt_binaries)
datas.extend(vtk_datas)
datas.extend(pyqt_datas)
hiddenimports.extend(vtk_hiddenimports)
hiddenimports.extend(pyqt_hiddenimports)

# ------------------------------------------------------------
# 3. 执行分析
# ------------------------------------------------------------
a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ------------------------------------------------------------
# 4. 构建 EXE（保持不变）
# ------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AnklePro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 调试窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
