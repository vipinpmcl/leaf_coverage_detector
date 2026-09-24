from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QComboBox, QSlider, QToolBar, QVBoxLayout, QWidget,
    QCheckBox
)

from .model import LabelMeDocument
from .canvas import ImageCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LabelMe Magnet Editor")
        self.resize(1400, 900)

        self.document = LabelMeDocument()
        self.undo_stack = []
        self.redo_stack = []
        self.stroke_before = None
        self.current_json_path = None

        self.canvas = ImageCanvas(self)
        self.setCentralWidget(self.canvas)

        self._build_toolbar()
        self.canvas.stroke_started.connect(self._stroke_started)
        self.canvas.points_changed.connect(self._stroke_finished)
        self.statusBar().showMessage("Open a LabelMe JSON file")

    def _build_toolbar(self):
        toolbar = QToolBar("Tools")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open JSON", self)
        open_action.triggered.connect(self.open_dialog)
        toolbar.addAction(open_action)

        save_action = QAction("Save As", self)
        save_action.triggered.connect(self.save_as)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        undo = QAction("Undo", self)
        undo.setShortcut("Ctrl+Z")
        undo.triggered.connect(self.undo)
        toolbar.addAction(undo)

        redo = QAction("Redo", self)
        redo.setShortcut("Ctrl+Y")
        redo.triggered.connect(self.redo)
        toolbar.addAction(redo)

        toolbar.addSeparator()

        fit = QAction("Fit", self)
        fit.triggered.connect(self.canvas.reset_view)
        toolbar.addAction(fit)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  Mode: "))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Attract", "Repel", "Erase"])
        self.mode_combo.currentTextChanged.connect(
            lambda x: setattr(self.canvas, "mode", x.lower())
        )
        toolbar.addWidget(self.mode_combo)

        toolbar.addWidget(QLabel("  Force: "))
        self.force_slider = QSlider(Qt.Orientation.Horizontal)
        self.force_slider.setRange(1, 100)
        self.force_slider.setValue(45)
        self.force_slider.setFixedWidth(130)
        self.force_slider.valueChanged.connect(self._force_changed)
        toolbar.addWidget(self.force_slider)
        self.force_label = QLabel("0.45")
        toolbar.addWidget(self.force_label)

        toolbar.addWidget(QLabel("  Size: "))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(5, 500)
        self.size_slider.setValue(60)
        self.size_slider.setFixedWidth(150)
        self.size_slider.valueChanged.connect(self._size_changed)
        toolbar.addWidget(self.size_slider)
        self.size_label = QLabel("60 px")
        toolbar.addWidget(self.size_label)

        toolbar.addSeparator()

        self.poly_check = QCheckBox("Polygon")
        self.poly_check.setChecked(True)
        self.poly_check.toggled.connect(self._polygon_toggled)
        toolbar.addWidget(self.poly_check)

        self.brush_check = QCheckBox("Brush")
        self.brush_check.setChecked(True)
        self.brush_check.toggled.connect(self._brush_toggled)
        toolbar.addWidget(self.brush_check)

    def _force_changed(self, value):
        force = value / 100.0
        self.canvas.force = force
        self.force_label.setText(f"{force:.2f}")

    def _size_changed(self, value):
        self.canvas.radius = float(value)
        self.canvas.eraser_radius = float(value)
        self.size_label.setText(f"{value} px")
        self.canvas.viewport().update()

    def _polygon_toggled(self, value):
        self.canvas.show_polygons = value
        self.canvas.viewport().update()

    def _brush_toggled(self, value):
        self.canvas.show_brush = value
        self.canvas.viewport().update()

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open LabelMe JSON", "", "LabelMe JSON (*.json)"
        )
        if path:
            self.open_json(Path(path))

    def open_json(self, path: Path):
        try:
            self.document.load(path)
            self.current_json_path = path
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.canvas.set_document(self.document.image_bgr, self.document.shapes)

            count = sum(
                1 for s in self.document.shapes
                if s.get("shape_type", "polygon") == "polygon"
            )
            self.statusBar().showMessage(
                f"Loaded: {path.name} | polygon shapes: {count} | "
                f"image: {self.document.image_path.name}"
            )
            self.setWindowTitle(f"LabelMe Magnet Editor — {path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))

    def _stroke_started(self):
        self.stroke_before = self.document.copy_points()

    def _stroke_finished(self):
        if self.stroke_before is not None:
            after = self.document.copy_points()
            if after != self.stroke_before:
                self.undo_stack.append((self.stroke_before, after))
                self.redo_stack.clear()
            self.stroke_before = None

    def undo(self):
        if not self.undo_stack:
            return
        before, after = self.undo_stack.pop()
        self.redo_stack.append((before, after))
        self.document.restore_points(before)
        self.canvas.viewport().update()
        self.statusBar().showMessage("Undo")

    def redo(self):
        if not self.redo_stack:
            return
        before, after = self.redo_stack.pop()
        self.undo_stack.append((before, after))
        self.document.restore_points(after)
        self.canvas.viewport().update()
        self.statusBar().showMessage("Redo")

    def save_as(self):
        if not self.current_json_path:
            QMessageBox.information(self, "Save", "Open a LabelMe JSON first.")
            return

        default = self.current_json_path.with_name(
            self.current_json_path.stem + "_edited.json"
        )

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Edited LabelMe JSON", str(default),
            "LabelMe JSON (*.json)"
        )
        if not path:
            return

        try:
            out = self.document.save(Path(path))
            self.statusBar().showMessage(f"Saved: {out}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def closeEvent(self, event):
        if self.undo_stack:
            answer = QMessageBox.question(
                self,
                "Exit",
                "You have unsaved edits. Exit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        event.accept()
