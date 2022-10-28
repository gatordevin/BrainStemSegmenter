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


# Subclass QMainWindow to customize your application's main window
class WelcomePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        center_msg = QLabel("Welcome")
        center_msg.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        layout.addWidget(center_msg)

        dataset_options = QHBoxLayout()
        create_dataset_button = QPushButton("Create Dataset")
        create_dataset_button.clicked.connect(lambda: self.buttonClick(button))
        dataset_options.addWidget(create_dataset_button)
        load_dataset_button = QPushButton("Load Dataset")
        load_dataset_button.clicked.connect(lambda: self.buttonClick(button))
        dataset_options.addWidget(load_dataset_button)
        dataset_options.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        layout.addLayout(dataset_options)
        