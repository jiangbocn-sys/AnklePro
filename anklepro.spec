# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置
使用方法:
    pyinstaller anklepro.spec

输出:
    dist/AnklePro.exe - Windows 可执行文件
"""

block_cipher = None

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src/config.py', 'src'),
        ('src/core', 'src/core'),
        ('src/render', 'src/render'),
        ('src/ui', 'src/ui'),
    ],
    hiddenimports=[
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
    ],
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
    console=True,  # 改为 True 显示调试窗口，便于排查问题
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
