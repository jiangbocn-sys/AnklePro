"""STEP ↔ STL 转换服务 — 通过 FreeCAD CLI 实现"""

import os
import subprocess
import tempfile
from typing import Optional


FREECAD_CMD = "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd"

# FreeCAD Python 脚本模板
_STEP_TO_STL_SCRIPT = """
import Part, os, FreeCAD, sys

step_path = {step_path!r}
stl_path = {stl_path!r}
linear_deflection = {linear_deflection!r}
angular_deflection = {angular_deflection!r}

try:
    shape = Part.read(step_path)
    doc = FreeCAD.newDocument('StepConvert')
    obj = doc.addObject('Part::Feature', 'Shape')
    obj.Shape = shape
    doc.recompute()

    # 设置网格化参数
    if hasattr(shape, 'tessellate'):
        shape.tessellate(linear_deflection)

    Part.export([obj], stl_path)
    size = os.path.getsize(stl_path)
    print(f'{{len(shape.Faces)}} faces, {{size}} bytes')
except Exception as e:
    print(f'ERROR: {{e}}', file=sys.stderr)
    sys.exit(1)
finally:
    try:
        FreeCAD.closeDocument(doc.Name)
    except Exception:
        pass
"""

_STL_TO_STEP_SCRIPT = """
import Part, os, FreeCAD, Mesh, sys

stl_path = {stl_path!r}
step_path = {step_path!r}
tolerance = {tolerance!r}

try:
    mesh = Mesh.Mesh()
    mesh.read(stl_path)
    shape = Part.Shape()
    shape.makeShapeFromMesh(mesh.Topology, tolerance)
    doc = FreeCAD.newDocument('StlConvert')
    obj = doc.addObject('Part::Feature', 'Shape')
    obj.Shape = shape
    doc.recompute()
    Part.export([obj], step_path)
    size = os.path.getsize(step_path)
    print(f'{{len(shape.Faces)}} faces, {{size}} bytes')
except Exception as e:
    print(f'ERROR: {{e}}', file=sys.stderr)
    sys.exit(1)
finally:
    try:
        FreeCAD.closeDocument(doc.Name)
    except Exception:
        pass
"""

_GET_INFO_SCRIPT = """
import Part, os, FreeCAD, sys

step_path = {step_path!r}

try:
    shape = Part.read(step_path)
    bounds = shape.BoundingBox
    info = dict(
        faces=len(shape.Faces),
        edges=len(shape.Edges),
        shells=len(shape.Shells),
        solids=len(shape.Solids),
        volume=shape.Volume if hasattr(shape, 'Volume') else 0.0,
        x_min=bounds.XMin, x_max=bounds.XMax,
        y_min=bounds.YMin, y_max=bounds.YMax,
        z_min=bounds.ZMin, z_max=bounds.ZMax,
    )
    for k, v in info.items():
        print(f'{{k}}={{v}}')
except Exception as e:
    print(f'ERROR: {{e}}', file=sys.stderr)
    sys.exit(1)
"""


def _run_freecad(script: str, timeout: int = 1800) -> str:
    """执行 FreeCAD CLI 脚本，返回 stdout"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(script)
        f.flush()
        script_path = f.name

    try:
        result = subprocess.run(
            [FREECAD_CMD, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FreeCAD 执行失败 (code={result.returncode}):\n"
                f"stderr: {result.stderr[-2000:]}"
            )
        return result.stdout
    finally:
        os.unlink(script_path)


def _parse_info_output(stdout: str) -> dict:
    """解析 FreeCAD 输出的 key=value 行"""
    info = {}
    for line in stdout.strip().splitlines():
        if "=" in line and not line.startswith("ERROR"):
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            try:
                if "." in value:
                    info[key] = float(value)
                else:
                    info[key] = int(value)
            except ValueError:
                info[key] = value
    return info


def step_to_stl(
    step_path: str,
    stl_path: Optional[str] = None,
    linear_deflection: float = 0.05,
    angular_deflection: float = 0.5,
) -> str:
    """
    将 STEP 文件转换为 STL

    参数:
        step_path: STEP 文件路径
        stl_path: 输出 STL 路径（默认与 STEP 同目录同名）
        linear_deflection: 线性偏差阈值 (mm)，越小网格越精细
        angular_deflection: 角度偏差阈值 (度)

    返回:
        输出 STL 文件路径
    """
    step_path = os.path.abspath(step_path)
    if stl_path is None:
        stl_path = os.path.splitext(step_path)[0] + ".stl"
    else:
        stl_path = os.path.abspath(stl_path)

    script = _STEP_TO_STL_SCRIPT.format(
        step_path=step_path,
        stl_path=stl_path,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )

    _run_freecad(script, timeout=3600)
    return stl_path


def stl_to_step(
    stl_path: str,
    step_path: Optional[str] = None,
    tolerance: float = 0.1,
) -> str:
    """
    将 STL 文件转换为 STEP（近似曲面重建）

    注意：离散网格转 BREP 会损失精度，仅作参考

    参数:
        stl_path: STL 文件路径
        step_path: 输出 STEP 路径（默认与 STL 同目录同名）
        tolerance: 网格合并容差 (mm)

    返回:
        输出 STEP 文件路径
    """
    stl_path = os.path.abspath(stl_path)
    if step_path is None:
        step_path = os.path.splitext(stl_path)[0] + ".step"
    else:
        step_path = os.path.abspath(step_path)

    script = _STL_TO_STEP_SCRIPT.format(
        stl_path=stl_path,
        step_path=step_path,
        tolerance=tolerance,
    )

    _run_freecad(script)
    return step_path


def get_step_info(step_path: str) -> dict:
    """
    获取 STEP 文件几何信息

    返回:
        {faces, edges, shells, solids, volume, x_min, x_max, y_min, y_max, z_min, z_max}
    """
    step_path = os.path.abspath(step_path)
    script = _GET_INFO_SCRIPT.format(step_path=step_path)
    stdout = _run_freecad(script)
    return _parse_info_output(stdout)
