"""全局配置"""

import numpy as np


def generate_gradient_color(value: float, min_val: float, max_val: float,
                            start_color: tuple, end_color: tuple) -> tuple:
    """
    根据值在范围内的位置生成渐变颜色

    Args:
        value: 当前值
        min_val: 范围最小值
        max_val: 范围最大值
        start_color: 起始颜色 (R, G, B)
        end_color: 结束颜色 (R, G, B)

    Returns:
        插值颜色 (R, G, B)
    """
    if max_val <= min_val:
        return start_color
    t = (value - min_val) / (max_val - min_val)
    t = max(0, min(1, t))  # 限制在 [0, 1] 范围
    r = start_color[0] + t * (end_color[0] - start_color[0])
    g = start_color[1] + t * (end_color[1] - start_color[1])
    b = start_color[2] + t * (end_color[2] - start_color[2])
    return (r, g, b)


def generate_16_level_thresholds():
    """
    生成 16 度渐进色阈值表（每 1mm 一级）

    颜色方案:
    - 负值到 0: 深红 → 鲜红 (穿透 → 接触)
    - 1-3mm: 鲜红 → 橙红 (偏紧过渡到理想)
    - 4-5mm: 绿色 (理想)
    - 6-8mm: 黄绿 → 黄色 (微松)
    - 9-11mm: 黄色 → 橙黄 (偏松)
    - 12-14mm: 橙黄 → 深黄色 (过松)
    - >=15mm: 深黄色 (严重过松)
    """
    thresholds = []

    # 负值区域：深红 → 鲜红 (每 1mm 一级)
    deep_red = (0.55, 0.0, 0.0)    # 深红
    bright_red = (1.0, 0.0, 0.0)   # 鲜红

    thresholds.append((float("-inf"), -5.0, deep_red, "< -5mm (严重穿透)"))
    for i in range(-5, 0):
        color = generate_gradient_color(i, -5, 0, deep_red, bright_red)
        thresholds.append((i, i + 1, color, f"{i} ~ {i+1}mm"))

    # 0-3mm: 鲜红 → 橙红 → 橙 (偏紧过渡到理想)
    orange_red = (1.0, 0.4, 0.0)   # 橙红
    orange = (1.0, 0.55, 0.0)      # 橙
    for i in range(0, 4):
        color = generate_gradient_color(i, 0, 4, bright_red, (0.0, 1.0, 0.0))
        thresholds.append((i, i + 1, color, f"{i} ~ {i+1}mm"))

    # 4-5mm: 绿色 (理想)
    thresholds.append((4.0, 5.0, (0.0, 1.0, 0.0), "4 ~ 5mm (理想)"))
    thresholds.append((5.0, 6.0, (0.15, 0.95, 0.15), "5 ~ 6mm (理想)"))

    # 6-8mm: 绿 → 黄绿 → 黄 (微松)
    yellow_green = (0.7, 1.0, 0.0)  # 黄绿
    yellow = (1.0, 1.0, 0.0)        # 黄
    for i in range(6, 9):
        color = generate_gradient_color(i, 6, 9, (0.3, 0.9, 0.3), yellow)
        thresholds.append((i, i + 1, color, f"{i} ~ {i+1}mm"))

    # 9-11mm: 黄 → 橙黄 (偏松)
    orange_yellow = (1.0, 0.8, 0.0)  # 橙黄
    for i in range(9, 12):
        color = generate_gradient_color(i, 9, 12, yellow, (1.0, 0.7, 0.0))
        thresholds.append((i, i + 1, color, f"{i} ~ {i+1}mm"))

    # 12-14mm: 橙黄 → 深黄色 (过松)
    deep_yellow = (0.85, 0.65, 0.0)  # 深黄
    for i in range(12, 15):
        color = generate_gradient_color(i, 12, 15, (1.0, 0.7, 0.0), deep_yellow)
        thresholds.append((i, i + 1, color, f"{i} ~ {i+1}mm"))

    # >=15mm: 深黄色 (严重过松)
    thresholds.append((15.0, float("inf"), deep_yellow, "≥ 15mm (严重过松)"))

    return thresholds


# ============================================================
# 颜色阈值配置（有符号距离，单位 mm）
# 负值 = 穿透 (护具在足部内部)，正值 = 间隙
# 格式：(min, max, (R, G, B), label)
# ============================================================
PRESET_THRESHOLDS = {
    "16 度渐进色": generate_16_level_thresholds(),
    "默认 (4mm 软垫)": [
        (float("-inf"), -0.1, (0.7, 0.0, 0.0), "穿透 (深红)"),
        (-0.1, 0.0,    (1.0, 0.3, 0.3), "接触 (浅红)"),
        (0.0, 4.0,     (1.0, 0.0, 0.0), "偏紧 (红)"),
        (4.0, 6.0,     (0.0, 1.0, 0.0), "理想 (绿)"),
        (6.0, 7.0,     (1.0, 1.0, 0.0), "偏松 (黄)"),
        (7.0, float("inf"), (1.0, 0.6, 0.0), "过松 (橙)"),
    ],
    "薄软垫 (2mm)": [
        (float("-inf"), -0.1, (0.7, 0.0, 0.0), "穿透 (深红)"),
        (-0.1, 0.0,    (1.0, 0.3, 0.3), "接触 (浅红)"),
        (0.0, 2.0,     (1.0, 0.0, 0.0), "偏紧 (红)"),
        (2.0, 4.0,     (0.0, 1.0, 0.0), "理想 (绿)"),
        (4.0, 5.0,     (1.0, 1.0, 0.0), "偏松 (黄)"),
        (5.0, float("inf"), (1.0, 0.6, 0.0), "过松 (橙)"),
    ],
    "厚软垫 (6mm)": [
        (float("-inf"), -0.1, (0.7, 0.0, 0.0), "穿透 (深红)"),
        (-0.1, 0.0,    (1.0, 0.3, 0.3), "接触 (浅红)"),
        (0.0, 6.0,     (1.0, 0.0, 0.0), "偏紧 (红)"),
        (6.0, 8.0,     (0.0, 1.0, 0.0), "理想 (绿)"),
        (8.0, 9.0,     (1.0, 1.0, 0.0), "偏松 (黄)"),
        (9.0, float("inf"), (1.0, 0.6, 0.0), "过松 (橙)"),
    ],
}

DEFAULT_PRESET = "16 度渐进色"

# ============================================================
# 交互设置
# ============================================================
DEFAULT_MOVE_STEP = 1.0        # 默认移动步长 (mm)
FINE_MOVE_STEP = 0.1           # 精细移动步长 (mm)
DEFAULT_ROTATE_STEP = 1.0      # 默认旋转步长 (度)
FINE_ROTATE_STEP = 0.1         # 精细旋转步长 (度)

# ============================================================
# 渲染设置
# ============================================================
BACKGROUND_COLOR = (0.1, 0.1, 0.15)   # 深灰蓝
FOOT_COLOR = (0.65, 0.65, 0.7)        # 足部默认颜色 (灰蓝)
BRACE_COLOR = (0.85, 0.75, 0.55)      # 护具默认颜色 (米黄)
MODEL_OPACITY_NORMAL = 0.7            # 正常模式透明度
MODEL_OPACITY_RESULT = 0.85           # 结果模式透明度
AXES_LENGTH = 50.0                    # 坐标轴长度 (mm)

# ============================================================
# 性能设置
# ============================================================
KD_TREE_DOWNSAMPLE = 8         # KD-tree 构建时的降采样因子
REALTIME_THROTTLE_MS = 200     # 实时计算节流间隔 (ms)
REALTIME_MOVE_THRESHOLD = 0.5  # 触发实时计算的最小位移 (mm)

# ============================================================
# 表面选取设置
# ============================================================
DEFAULT_NORMAL_ANGLE_THRESHOLD = 60.0  # 法向筛选角度阈值 (度)
