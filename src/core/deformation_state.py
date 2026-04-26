"""变形状态管理 — 记录变形历史、支持撤销、持久化"""

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np

from src.core.deformation_engine import DeformationParams


@dataclass
class DeformationStep:
    """单步变形记录"""
    mode: str
    offset_mm: float
    scale_factor: float
    decay_radius: float
    direction: Optional[List[float]]
    center_point: Optional[List[float]]
    region_indices: List[int]  # JSON 可序列化的索引列表


class DeformationState:
    """
    变形历史管理

    记录每一步变形参数，支持撤销操作，可将变形历史保存/加载为 JSON。
    """

    def __init__(self):
        self._history: List[DeformationStep] = []

    @property
    def step_count(self) -> int:
        return len(self._history)

    def push(self, step: DeformationStep):
        """推入一步变形记录"""
        self._history.append(step)

    def undo(self) -> Optional[DeformationStep]:
        """撤销上一步变形，返回被撤销的步骤"""
        if not self._history:
            return None
        return self._history.pop()

    def clear(self):
        """清空所有变形历史"""
        self._history.clear()

    def get_params_list(self) -> List[DeformationParams]:
        """将历史记录转为 DeformationParams 列表（用于重新计算）"""
        params_list = []
        for step in self._history:
            direction = (
                np.array(step.direction) if step.direction else None
            )
            center = (
                np.array(step.center_point) if step.center_point else None
            )
            params_list.append(DeformationParams(
                mode=step.mode,
                region_indices=np.array(step.region_indices, dtype=np.int64),
                offset_mm=step.offset_mm,
                scale_factor=step.scale_factor,
                decay_radius=step.decay_radius,
                direction=direction,
                center_point=center,
            ))
        return params_list

    def save(self, filepath: str):
        """保存变形参数到 JSON"""
        data = {
            "steps": [asdict(s) for s in self._history],
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "DeformationState":
        """从 JSON 加载变形参数"""
        state = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for step_data in data.get("steps", []):
            state._history.append(DeformationStep(**step_data))

        return state
