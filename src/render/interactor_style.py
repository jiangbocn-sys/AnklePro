"""自定义 VTK 交互器 — 鼠标控制相机视角，键盘控制护具变换"""

import vtk


class BraceCameraInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """
    自定义 VTK 交互器样式

    - 鼠标左键拖拽 → 旋转相机视角
    - 鼠标右键拖拽 → 平移相机
    - 鼠标滚轮 → 缩放
    - 测量模式下鼠标事件通过 Qt 事件过滤器处理
    - 键盘控制护具变换通过 Qt 事件过滤器处理
    """

    def __init__(self, renderer=None, main_window=None):
        super().__init__()
        self.renderer = renderer
        self.main_window = main_window

    def set_measuring(self, enabled: bool):
        """开启/关闭测量模式"""
        pass

    def OnKeyDown(self):
        """处理键盘按下事件"""
        key = self.GetInteractor().GetKeySym()

        # 按 D 键切换线框/实体模式
        if key.lower() == 'd':
            if self.main_window:
                self.main_window._toggle_wireframe()

        super().OnKeyDown()
