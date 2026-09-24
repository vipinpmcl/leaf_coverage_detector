import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from .window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LabelMe Magnet Editor")
    app.setOrganizationName("LeafCoverageDetector")
    window = MainWindow()

    if len(sys.argv) > 1:
        window.open_json(Path(sys.argv[1]))

    window.show()
    sys.exit(app.exec())
