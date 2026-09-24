# LabelMe Magnet Editor

A lightweight PySide6 desktop editor for LabelMe polygon annotations.

## Features
- Open LabelMe JSON directly (and its image)
- Display polygon overlays on the original image
- Zoom and pan
- Attract / Repel magnet brush
- Eraser brush for deleting polygon vertices or an entire polygon
- Adjustable brush size and force
- Configurable eraser size
- Smooth distance falloff
- Undo / Redo
- Save edited annotation as a new LabelMe JSON
- Preserves LabelMe metadata and non-polygon shapes

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Or open a JSON directly:

```bash
python main.py path/to/image.json
```

## Notes

The magnet operates on the existing polygon vertices; it does not create subdivision vertices.
For a 4-point rectangle, only those 4 points can move, so producing a smooth circle requires a polygon that already has enough vertices. This implementation intentionally follows the requested V1 + V4 design and does not add a subdivision/edit-vertex tool.

## Eraser behavior

Select **Erase** mode and drag over polygon vertices. Vertices inside the eraser radius are removed. If a polygon is reduced below three vertices, it is removed. Undo/Redo restores the previous state.
