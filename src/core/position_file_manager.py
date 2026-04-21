"""护具位置文件管理 — 保存和加载护具相对位置"""

import json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np


class PositionFileManager:
    """
    护具位置文件管理器

    功能：
    1. 自动根据护具 STL 文件路径生成位置文件路径
    2. 保存位置到 JSON 文件
    3. 从 JSON 文件加载位置
    """

    def __init__(self):
        # 位置文件存储在用户主目录的 .anklepro 文件夹
        self.base_dir = Path.home() / ".anklepro" / "positions"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_position_file_path(self, stl_file_path: str) -> Path:
        """
        根据 STL 文件路径生成位置文件路径

        规则：
        - 使用 STL 文件名的 MD5 哈希作为文件名
        - 避免路径和特殊字符问题

        参数:
            stl_file_path: STL 文件的完整路径

        返回:
            位置文件的完整路径
        """
        import hashlib

        # 使用文件路径的 MD5 哈希作为文件名
        path_hash = hashlib.md5(stl_file_path.encode('utf-8')).hexdigest()
        return self.base_dir / f"{path_hash}.json"

    def save_positions(
        self,
        stl_file_path: str,
        positions: List[Optional[Tuple[List[float], List[float]]]],
        inner_surface_indices: Optional[np.ndarray] = None,
        inner_regions: Optional[List[np.ndarray]] = None,
    ):
        """
        保存护具位置到文件

        参数:
            stl_file_path: STL 文件路径
            positions: 位置列表，每个元素为 (translation, rotation) 元组
                      translation 和 rotation 为 numpy 数组
            inner_surface_indices: 内侧面顶点索引 numpy 数组
            inner_regions: 内侧面连通区域列表（每个区域为 numpy 数组）
        """
        file_path = self._get_position_file_path(stl_file_path)

        # 转换内侧面数据为 JSON 可序列化格式
        indices_data = None
        if inner_surface_indices is not None:
            if isinstance(inner_surface_indices, np.ndarray):
                indices_data = inner_surface_indices.tolist()
            else:
                indices_data = inner_surface_indices

        regions_data = None
        if inner_regions is not None:
            regions_data = []
            for region in inner_regions:
                if isinstance(region, np.ndarray):
                    regions_data.append(region.tolist())
                else:
                    regions_data.append(region)

        data = {
            "stl_file": stl_file_path,
            "positions": [],
            "inner_surface_indices": indices_data,
            "inner_regions": regions_data,
        }

        # 转换位置数据为 JSON 可序列化格式
        for i, pos in enumerate(positions):
            if pos is None:
                data["positions"].append(None)
            else:
                translation, rotation = pos
                # 确保是列表格式
                if isinstance(translation, np.ndarray):
                    translation = translation.tolist()
                if isinstance(rotation, np.ndarray):
                    rotation = rotation.tolist()
                data["positions"].append({
                    "slot": i,
                    "translation": [float(x) for x in translation],
                    "rotation": [float(x) for x in rotation],
                })

        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"位置已保存到：{file_path}")

    def load_positions(self, stl_file_path: str) -> dict:
        """
        从文件加载护具位置

        参数:
            stl_file_path: STL 文件路径

        返回:
            包含以下键的字典：
            - positions: 位置列表
            - inner_surface_indices: 内侧面顶点索引
            - inner_regions: 内侧面连通区域
            - found: 是否找到位置文件
        """
        file_path = self._get_position_file_path(stl_file_path)

        if not file_path.exists():
            return {
                "positions": [None] * 5,
                "inner_surface_indices": None,
                "inner_regions": None,
                "found": False,
            }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 转换位置数据回 numpy 数组格式
            positions = [None] * 5
            for pos_data in data.get("positions", []):
                if pos_data is not None:
                    slot = pos_data.get("slot", 0)
                    if 0 <= slot < 5:
                        positions[slot] = (
                            np.array(pos_data["translation"], dtype=np.float64),
                            np.array(pos_data["rotation"], dtype=np.float64),
                        )

            return {
                "positions": positions,
                "inner_surface_indices": data.get("inner_surface_indices"),
                "inner_regions": data.get("inner_regions"),
                "found": True,
                "file_path": str(file_path),
            }

        except Exception as e:
            print(f"加载位置文件失败：{e}")
            return {
                "positions": [None] * 5,
                "inner_surface_indices": None,
                "inner_regions": None,
                "found": False,
                "error": str(e),
            }

    def delete_positions(self, stl_file_path: str) -> bool:
        """
        删除护具位置文件

        参数:
            stl_file_path: STL 文件路径

        返回:
            是否成功删除
        """
        file_path = self._get_position_file_path(stl_file_path)

        if file_path.exists():
            file_path.unlink()
            print(f"位置文件已删除：{file_path}")
            return True
        return False

    def list_all_positions(self) -> list:
        """
        列出所有已保存的位置文件

        返回:
            位置文件信息列表
        """
        files = []
        for file_path in self.base_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                files.append({
                    "file": str(file_path),
                    "stl_file": data.get("stl_file", "未知"),
                    "positions_count": sum(1 for p in data.get("positions", []) if p is not None),
                })
            except Exception as e:
                files.append({
                    "file": str(file_path),
                    "error": str(e),
                })
        return files
