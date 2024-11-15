# -*- coding: utf-8 -*-

# Created by: PyQt5 UI code generator 5.15.4

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout


class Ui_Form(QWidget):
    def __init__(self, parent=None):
        super(Ui_Form, self).__init__(parent)
        self.setWindowTitle("Input Name")
        # Light background color
        self.setStyleSheet("background-color: #f0f4f8;")
        self.resize(473, 362)

        # Set window icon
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon/warning.png"),
                       QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        # Create layouts
        self.main_layout = QVBoxLayout()
        self.label = QLabel("Please enter your name:", self)
        self.label.setStyleSheet("font-size: 18px; color: #34495e;")
        self.main_layout.addWidget(self.label)

        # Input field
        self.lineEdit = QLineEdit(self)
        self.lineEdit.setPlaceholderText("Enter your name")
        self.lineEdit.setStyleSheet(
            "border: 1px solid #bdc3c7; border-radius: 15px; padding: 10px; font-size: 16px;")
        self.main_layout.addWidget(self.lineEdit)

        # Button
        self.pushButton = QPushButton("Confirm", self)
        self.pushButton.setStyleSheet(
            "background-color: #3498db; color: white; border-radius: 15px; font-size: 18px; padding: 12px;")
        # Change cursor to pointer
        self.pushButton.setCursor(QtCore.Qt.PointingHandCursor)
        self.pushButton.clicked.connect(self.click)
        self.main_layout.addWidget(self.pushButton)

        # Set layout for the widget
        self.setLayout(self.main_layout)

    def click(self):
        name = self.lineEdit.text()
        import face_capture
        face_capture.read(name)
        self.close()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Ui_Form()
    window.show()
    sys.exit(app.exec_())
