# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AnklePro** — Interactive 3D visualization tool for analyzing ankle brace-to-foot fit. Built with PyQt5 + VTK + NumPy + SciPy.

The app loads foot and brace 3D models (STL format), computes signed distances between brace inner surface and foot surface, and visualizes fit quality via color mapping.

## Key Commands

```bash
# Run the application
python src/main.py

# Build Windows executable (via PyInstaller)
pyinstaller --clean --additional-hooks-dir=hooks anklepro.spec

# Or use the GitHub Actions workflow (push to main or tag)
```

Dependencies: `pip install -r requirements.txt` (PyQt5, VTK, NumPy, SciPy, matplotlib)

## Architecture

### Package Structure

```
src/
  main.py                  # Entry point — creates QApplication + MainWindow
  config.py                # Global config: color thresholds, presets, step sizes
  core/
    model_loader.py        # STL -> vtkPolyData (clean + normals + centroid)
    transform_manager.py   # 4x4 transform (separate rotation + translation, undo stack)
    distance_calculator.py # Signed distance via vtkCellLocator + vtkOBBTree ray casting
    surface_picker.py      # Inner surface identification (normal-based) + connected regions
    optimizer.py           # Grid search optimizer for best-fit position (radial distance)
    radial_distance_calculator.py  # Mid-axis based radial gap calculation
    position_file_manager.py       # JSON-based position save/load (~/.anklepro/positions/)
  render/
    scene_manager.py       # VTK scene: foot/brace actors, axes, scalar bar, color mapping
    interactor_style.py    # Custom VTK interactor (trackball camera)
  ui/
    main_window.py         # Main window: VTK view + control panel (tabs), keyboard handling
```

### Data Flow

1. **Load models**: `load_stl()` → `ModelData` (vtkPolyData + metadata)
2. **Inner surface pick**: `identify_inner_surface()` → connected regions → user selects regions
3. **Transform brace**: `TransformManager` applies translation/rotation via `vtkTransformPolyDataFilter`
4. **Compute distance**: `DistanceCalculator.compute_signed_distances()` → ray-casting inside/outside test
5. **Color map**: `SceneManager.apply_distance_colors()` maps distances to vertex colors via lookup table

### Distance Computation

Two methods available:
- **Signed distance** (`DistanceCalculator`): vtkCellLocator finds closest point, vtkOBBTree ray casting for inside/outside test. Positive = gap, negative = penetration.
- **Radial distance** (`RadialDistanceCalculator`): Projects onto mid-axis (PCA-fitted or Z-axis), computes radial gap = d_brace - d_foot. Used by optimizer to avoid inside/outside ambiguity.

### Keyboard Controls

- Arrow keys: camera zoom/rotate
- F/G: Z-axis rotation, H/J: Y-axis, N/M: X-axis
- B/V: X translation, U/I: Y translation, R/T: Z translation
- D: wireframe toggle, Ctrl+Z: undo

### Color Threshold Presets

Defined in `config.py`: "16 度渐进色" (default, 16-level gradient), "默认 (4mm 软垫)", "薄软垫 (2mm)", "厚软垫 (6mm)". Default scheme is "16 度渐进色".

### Build / Packaging

- PyInstaller spec: `anklepro.spec` — outputs `dist/AnklePro.exe`
- GitHub Actions: `.github/workflows/build-windows.yml` — builds on push to main or tag
- Custom hooks in `hooks/` directory for PyInstaller

## Important Notes

- `data/` directory contains test STL/STEP models (multi-GB, not in git)
- Development plan with 6 phases is documented in `DEVELOPMENT.md`
- The `stl_match_analysis.py` at repo root is a legacy offline analysis script, separate from the GUI app
- Windows build scripts (`build_windows.py`, `build.ps1`, etc.) exist but the CI workflow is the primary build method
