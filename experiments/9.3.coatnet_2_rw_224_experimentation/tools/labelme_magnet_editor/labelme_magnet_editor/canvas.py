from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QBrush
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from .brush import apply_magnet


class ImageCanvas(QGraphicsView):
    points_changed = Signal()
    stroke_started = Signal()
    cursor_image_position = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = None

        self.image_bgr = None
        self.shapes = []
        self.selected_shape = None

        self.mode = "attract"
        self.force = 0.45
        self.radius = 60.0
        self.eraser_radius = 30.0

        self.brushing = False
        self.last_scene_pos = None
        self.pan_active = False
        self.pan_start = None

        self.show_polygons = True
        self.show_brush = True
        self.brush_scene_pos = None

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setBackgroundBrush(QColor("#202020"))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)

    def set_document(self, image_bgr, shapes):
        self.image_bgr = image_bgr
        self.shapes = shapes
        self.selected_shape = None

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

        self.scene().clear()
        self.pixmap_item = QGraphicsPixmapItem()
        from PySide6.QtGui import QPixmap
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene().addItem(self.pixmap_item)
        self.scene().setSceneRect(0, 0, w, h)

        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.viewport().update()

    def image_to_scene(self, p):
        return QPointF(float(p[0]), float(p[1]))

    def scene_to_image(self, p):
        return np.array([p.x(), p.y()], dtype=np.float64)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.pan_active = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.pan_active = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        self.brush_scene_pos = scene_pos

        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and
            self.pan_active
        ):
            self.pan_active = True
            self.pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.show_polygons:
            self.stroke_started.emit()
            self.brushing = True
            self.last_scene_pos = scene_pos
            if self.mode == "erase":
                self._erase_at_position(scene_pos)
            else:
                self._apply_at_position(scene_pos)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        self.brush_scene_pos = scene_pos
        self.cursor_image_position.emit(scene_pos.x(), scene_pos.y())

        if self.pan_active and self.pan_start is not None:
            delta = event.position() - self.pan_start
            self.pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            self.viewport().update()
            event.accept()
            return

        if self.brushing and self.last_scene_pos is not None:
            # Interpolate brush positions so fast mouse movement doesn't skip strokes.
            a = self.last_scene_pos
            b = scene_pos
            dx, dy = b.x() - a.x(), b.y() - a.y()
            distance = (dx * dx + dy * dy) ** 0.5
            spacing = max(2.0, self.radius * 0.20)
            steps = max(1, int(distance / spacing))

            for i in range(1, steps + 1):
                t = i / steps
                p = QPointF(a.x() + dx * t, a.y() + dy * t)
                if self.mode == "erase":
                    self._erase_at_position(p)
                else:
                    self._apply_at_position(p)

            self.last_scene_pos = scene_pos
            event.accept()
            return

        self.viewport().update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton) and self.pan_active:
            self.pan_active = False
            self.pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.brushing:
            self.brushing = False
            self.last_scene_pos = None
            self.points_changed.emit()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.brush_scene_pos = None
        self.viewport().update()
        super().leaveEvent(event)

    def _apply_at_position(self, scene_pos):
        center = self.scene_to_image(scene_pos)

        # If no explicit selection exists, affect every polygon.
        # This is convenient for leaf annotations with a single polygon.
        for shape in self.shapes:
            if shape.get("shape_type", "polygon") != "polygon":
                continue
            points = shape.get("points", [])
            if len(points) < 2:
                continue

            new_points = apply_magnet(
                points, center, self.radius, self.force, self.mode
            )
            shape["points"] = new_points.tolist()

        self.viewport().update()

    def _erase_at_position(self, scene_pos):
        center = self.scene_to_image(scene_pos)
        radius = float(self.eraser_radius)
        new_shapes = []
        changed = False

        for shape in self.shapes:
            if shape.get("shape_type", "polygon") != "polygon":
                new_shapes.append(shape)
                continue

            points = np.asarray(shape.get("points", []), dtype=np.float64)
            if len(points) == 0:
                continue

            distances = np.linalg.norm(points - center[None, :], axis=1)
            kept = points[distances > radius]
            if len(kept) != len(points):
                changed = True

            if len(kept) >= 3:
                shape["points"] = kept.tolist()
                new_shapes.append(shape)
            elif len(kept) > 0:
                changed = True
                # Removing the last 1-2 vertices removes the invalid polygon.
            else:
                changed = True

        if changed:
            self.shapes[:] = new_shapes
        self.viewport().update()

    def reset_view(self):
        if self.pixmap_item:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)

        if self.show_polygons:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            for shape in self.shapes:
                if shape.get("shape_type", "polygon") != "polygon":
                    continue

                points = shape.get("points", [])
                if len(points) < 2:
                    continue

                qpoints = [QPointF(float(x), float(y)) for x, y in points]
                pen = QPen(QColor("#00ff66"), 2.0)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                for i in range(len(qpoints)):
                    painter.drawLine(qpoints[i], qpoints[(i + 1) % len(qpoints)])

                # Small vertices help verify the actual LabelMe points.
                painter.setBrush(QBrush(QColor("#ffcc00")))
                painter.setPen(QPen(QColor("#111111"), 1.0))
                for p in qpoints:
                    painter.drawEllipse(p, 3.0, 3.0)

            painter.restore()

        if self.show_brush and self.brush_scene_pos is not None:
            painter.save()
            p = self.brush_scene_pos
            radius = self.eraser_radius if self.mode == "erase" else self.radius

            color = QColor("#44aaff")
            color.setAlpha(180)
            painter.setPen(QPen(color, 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(p, radius, radius)

            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.drawLine(p.x() - 7, p.y(), p.x() + 7, p.y())
            painter.drawLine(p.x(), p.y() - 7, p.x(), p.y() + 7)
            painter.restore()
