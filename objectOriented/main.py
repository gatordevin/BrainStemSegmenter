import sys
from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
# from PySide2 import QtWidgets
# from PyQt5 import QtWidgets
from qt_material import apply_stylesheet
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QStackedLayout,
    QHBoxLayout,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from pages import *


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Widgets App")

        central = QWidget()
        pages = QStackedLayout()
        central.setLayout(pages)

        welcome_page = WelcomePage()
        pages.addWidget(welcome_page)
        pages.setCurrentIndex(0)

        self.setCentralWidget(central)

app = QtWidgets.QApplication(sys.argv)
apply_stylesheet(app, theme='dark_blue.xml')

window = MainWindow()
window.show()

app.exec()