# exol_3d_STLs - STL空间匹配分析项目

**创建日期**: 2026-04-15  
**项目目标**: 分析护具(exol)在足部(foot)上的最佳佩戴位置，验证3D模型生成算法

> **命名说明**: 
> - **exol** = 护具3D打印模型 (~28MB，网格密度超高)
> - **foot** = 足部扫描文件 (~5.2MB，扫描精度较低) 
> - **LEFT/RIGHT** = 左右脚

---

## 📁 目录结构

### STL模型文件 (10个)

| 文件名 | 说明 | 来源 |
|--------|------|------|
| `tangle_LEFT_exol.stl` | 左脚护具3D打印模型 (28MB) | CAD建模 Rhinoceros（超精密网格） |
| `tangle_LEFT_foot.stl` | 左脚足部扫描文件 (5.2MB) | 医疗扫描设备 |
| `tangle_RIGHT_exol.stl` | 右脚护具3D打印模型 (28MB) | CAD建模 |
| `tangle_RIGHT_foot.stl` | 右脚足部扫描文件 (5.2MB) | 医疗扫描设备 |
| `tangle_RIGHT_exol.stl` | 右脚护具3D打印模型 (5.2MB) | CAD建模 |
| `tangle_RIGHT_foot.stl` | 右脚足部扫描文件 (28MB) | 医疗扫描设备 |
| `tangle_RIGHT_exol.stl` | 右侧肢体扫描模型 | 医疗扫描设备 |
| `tangle_RIGHT_foot.stl` | 右侧护具模型 | CAD建模 |
| `qb_LEFT_exol.stl` | qb系列左侧肢体 | - |
| `qb_LEFT_foot.stl` | qb系列左侧护具 | - |
| `qb_RIGHT_exol.stl` | qb系列右侧肢体 | - |
| `qb_RIGHT_foot.stl` | qb系列右侧护具 | - |
| `2025-12-16_LEFT_brown.stl` | 2025年左侧模型 | Rhinoceros |
| `2025-12-16_RIGHT_brown.stl` | 2025年右侧模型 | Rhinoceros |

### 分析可视化 (12个PNG)

| 文件名 | 内容 | 关键发现 |
|--------|------|---------|
| `stl_equal_scale_final.png` | **最终等比例可视化** | 最佳匹配位置 |
| `stl_curve_matching_result.png` | 曲线匹配分析 | X宽度匹配99.6% |
| `stl_rotation_corrected.png` | 旋转修正对比 | 绕Z轴旋转180度 |
| `stl_wearing_validation.png` | 佩戴方向验证 | 确认正确方向 |
| `stl_matching_visualization.png` | 空间匹配可视化 | 初步分析 |
| `stl_final_best.png` | 最佳位置分析 | Y=55mm位置 |
| ... | ... | ... |

### 报告文档 (4个)

| 文件名 | 内容 |
|--------|------|
| `stl_analysis_report.txt` | STL文件解析分析 |
| `stl_comparison_report_tangle_LEFT.md` | tangle_LEFT对比报告 |
| `stl_matching_report.md` | 空间匹配分析报告 |
| `stl_wearing_validation_report.md` | 佩戴验证报告 |

### 分析脚本

| 文件名 | 功能 |
|--------|------|
| `stl_match_analysis.py` | 通用STL匹配分析工具 |

---

## 🎯 最终分析结果

### tangle_LEFT 最佳佩戴位置

**坐标系**: X=左右宽度, Y=前后长度(脚趾方向), Z=高度(脚底向上)

| 参数 | 值 | 说明 |
|------|-----|------|
| **X位置** | -7mm | 左右居中偏左 |
| **Y位置** | 57mm | 前后位置 |
| **Z位置** | 71mm | 高度（脚踝上方） |

### tangle_LEFT 形状匹配度

| 匹配项 | 比例 |
|--------|------|
| X宽度匹配 | 93.0% |
| Y长度匹配 | **98.4%** ✅ |
| 形状重合度 | **95.7%** ✅ |

### tangle_LEFT 贴合度验证

| 统计项 | 原始距离 | 加4mm软垫后 |
|--------|----------|-------------|
| 最小 | 0.06mm | -3.94mm |
| 平均 | 5.00mm | 1.00mm |
| 中位数 | **1.94mm** | -2.06mm |
| 完美贴合 | - | 10.8% |
| 过紧 | - | 61.5% ⚠️ |

---

### tangle_RIGHT 最佳佩戴位置

| 参数 | 值 | 说明 |
|------|-----|------|
| **X位置** | 6mm | 左右居中偏右 |
| **Y位置** | 57mm | 前后位置 |
| **Z位置** | 71mm | 高度（脚踝上方） |

### tangle_RIGHT 形状匹配度

| 匹配项 | 比例 |
|--------|------|
| X宽度匹配 | **99.4%** ✅ |
| Y长度匹配 | **99.9%** ✅ |
| 形状重合度 | **99.7%** ✅ ✅ |

### tangle_RIGHT 贴合度验证

| 统计项 | 原始距离 | 加4mm软垫后 |
|--------|----------|-------------|
| 最小 | 0.04mm | -3.96mm |
| 平均 | 5.33mm | 1.33mm |
| 中位数 | **3.83mm** | -0.17mm |
| 完美贴合 | - | 15.0% |
| 过紧 | - | 51.6% |

### 对比总结

| 对比项 | LEFT | RIGHT |
|--------|------|-------|
| 形状重合度 | 95.7% | **99.7%** ✅ |
| 中位数距离 | 1.94mm | 3.83mm |
| 完美贴合率 | 10.8% | **15.0%** |
| 过紧率 | 61.5% | **51.6%** |

**结论**: RIGHT模型匹配效果更好！

## 🔬 分析方法

### 1. 曲线匹配法
- 分析exol在不同位置的横截面曲线
- 比较foot内部曲线与exol曲线的形状
- 寻找X宽度、Y长度最匹配的位置

### 2. 旋转修正
- 绕foot自身Z轴(高度轴)旋转180度
- 使开口朝向脚趾方向(Y增大)

### 3. 等比例可视化
- `set_box_aspect([range_x, range_y, range_z])` - 3D等比例
- `set_aspect('equal')` - 2D截面等比例
- 各轴比例一致，无形变

---

## 🚀 使用方法

### 运行分析脚本

```bash
# 默认分析tangle_LEFT
python3 ~/.openclaw/workspace/exol_3d_STLs/stl_match_analysis.py

# 分析其他模型
python3 ~/.openclaw/workspace/exol_3d_STLs/stl_match_analysis.py qb_LEFT_exol.stl qb_LEFT_foot.stl
```

### 在Python中使用

```python
from stl_match_analysis import parse_binary_stl, rotate_z_180, find_best_match_position

exol = parse_binary_stl('tangle_LEFT_exol.stl')
foot = parse_binary_stl('tangle_LEFT_foot.stl')
foot_rotated = rotate_z_180(foot)

best = find_best_match_position(exol, foot_rotated)
```

---

## ⚠️ 发现的问题

### 1. 尺寸差异
- foot内部尺寸小于脚踝扫描尺寸
- 建议：放大foot模型约1.3倍

### 2. 过紧比例高
- 加4mm软垫后61.5%过紧
- 说明：foot可能偏小，或需要调整位置

### 3. 坐标系验证
- exol数据：Y=0~270mm(前后), Z=0~352mm(高度)
- Z=352mm较大，可能是躺着扫描或包含小腿

---

## 📊 关键发现总结

1. **正确佩戴方向**: 绕Z轴(高度轴)旋转180度，开口朝向Y增大(脚趾方向)
2. **最佳匹配位置**: X=-7mm, Y=57mm, Z=71mm
3. **形状匹配**: 95.7%重合度，Y长度匹配98.4%
4. **贴合质量**: 中位数距离1.94mm(较好)，但61.5%过紧需调整

---

*项目整理于 2026-04-15 | Jessie for 波哥*