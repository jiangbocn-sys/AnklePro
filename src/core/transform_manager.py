"""变换管理器 — 管理护具的4x4变换矩阵（平移、旋转、缩放）"""

import numpy as np


class TransformManager:
    """
    管理模型的4x4齐次变换矩阵

    内部将旋转和平移分开存储，确保：
    - 平移在世界空间中进行（沿XYZ轴移动）
    - 旋转在模型局部空间中进行（绕质心自转）
    - 两者互不干扰
    """

    def __init__(self, center: np.ndarray = None):
        self.center = np.zeros(3) if center is None else center.copy()
        self._rotation = np.eye(3, dtype=np.float64)  # 纯旋转 (3x3)
        self._translation = np.zeros(3, dtype=np.float64)  # 纯平移 (3,)
        self._history = []  # 撤销栈

    def translate(self, dx: float, dy: float, dz: float):
        """在世界空间中平移"""
        self._push_history()
        self._translation += np.array([dx, dy, dz], dtype=np.float64)

    def rotate_x(self, angle_deg: float):
        """绕模型质心旋转X轴"""
        self._rotate_local(_rotation_x(angle_deg))

    def rotate_y(self, angle_deg: float):
        """绕模型质心旋转Y轴"""
        self._rotate_local(_rotation_y(angle_deg))

    def rotate_z(self, angle_deg: float):
        """绕模型质心旋转Z轴"""
        self._rotate_local(_rotation_z(angle_deg))

    def apply(self, vertices: np.ndarray) -> np.ndarray:
        """对顶点数组应用当前变换: v' = R @ (v - center) + center + T"""
        if len(vertices) == 0:
            return vertices
        centered = vertices - self.center
        rotated = (self._rotation @ centered.T).T
        return rotated + self.center + self._translation

    def get_translation(self) -> np.ndarray:
        """获取当前平移量"""
        return self._translation.copy()

    def get_rotation(self) -> np.ndarray:
        """
        从旋转矩阵提取欧拉角 (度)

        使用 XYZ 顺序提取旋转角度
        返回：[rx, ry, rz] 单位为度

        注意：由于万向锁问题，提取的角度可能与输入角度不完全一致，
        但代表的旋转状态是等价的。
        """
        R = self._rotation

        # 防止数值误差导致超出定义域
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            # 标准情况
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            # 万向锁情况 (ry = ±90°)
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0

        return np.degrees([rx, ry, rz])

    def get_matrix(self) -> np.ndarray:
        """获取4x4齐次变换矩阵"""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self._rotation
        matrix[:3, 3] = self._translation + self.center - self._rotation @ self.center
        return matrix

    def reset(self):
        """重置为单位矩阵"""
        self._push_history()
        self._rotation = np.eye(3, dtype=np.float64)
        self._translation = np.zeros(3, dtype=np.float64)

    # ---- 撤销/重做 ----

    def undo(self) -> bool:
        if not self._history:
            return False
        rot, trans = self._history.pop()
        self._rotation = rot
        self._translation = trans
        return True

    def can_undo(self) -> bool:
        return len(self._history) > 0

    def _push_history(self):
        self._history.append((self._rotation.copy(), self._translation.copy()))
        if len(self._history) > 50:
            self._history = self._history[-50:]

    # ---- 内部方法 ----

    def _rotate_local(self, rotation: np.ndarray):
        """在局部空间中应用旋转（绕质心）"""
        self._push_history()
        self._rotation = self._rotation @ rotation


def _rotation_x(angle_deg: float) -> np.ndarray:
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float64)


def _rotation_y(angle_deg: float) -> np.ndarray:
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def _rotation_z(angle_deg: float) -> np.ndarray:
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=np.float64)
