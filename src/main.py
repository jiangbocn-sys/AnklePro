"""AnklePro — 足部-护具3D贴合度可视化分析工具

程序入口
"""

import sys
from pathlib import Path

# Add project root to path so 'src' package is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from src.ui.main_window import MainWindow


def main():
    # 高 DPI 支持（Windows）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("AnklePro")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("AnklePro")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
