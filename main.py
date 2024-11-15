from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QPushButton, QLabel
import model_train
import camera_reader

user = 'admin'

class Ui_Form(QWidget):
    def __init__(self, parent=None):
        super(Ui_Form, self).__init__(parent)
        self.setWindowTitle("Main Menu")
        self.setStyleSheet("background-color: #ffffff;")  # Clean white background
        self.resize(800, 600)

        # Create main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(QtCore.Qt.AlignCenter)  # Center content

        # Welcome label
        self.welcome_label = QLabel(f"Welcome, {user}!", self)
        self.welcome_label.setStyleSheet("font-size: 24px; color: #2C3E50;")
        self.main_layout.addWidget(self.welcome_label)

        # Create buttons
        self.create_buttons()

        # Set the main layout
        self.setLayout(self.main_layout)

    def create_buttons(self):
        button_style = (
            "QPushButton {"
            "background-color: #3498db; "
            "color: white; "
            "border-radius: 8px; "
            "font-size: 18px; "
            "padding: 15px; "
            "margin: 10px;"
            "}"
            "QPushButton:hover {"
            "background-color: #2980b9;"
            "}"
        )

        self.face_registration_button = QPushButton("Face Registration", self)
        self.face_registration_button.setStyleSheet(button_style)
        self.face_registration_button.clicked.connect(self.click1)
        self.main_layout.addWidget(self.face_registration_button)

        self.model_training_button = QPushButton("Model Training", self)
        self.model_training_button.setStyleSheet(button_style)
        self.model_training_button.clicked.connect(self.click2)
        self.main_layout.addWidget(self.model_training_button)

        self.face_recognition_button = QPushButton("Face Recognition", self)
        self.face_recognition_button.setStyleSheet(button_style)
        self.face_recognition_button.clicked.connect(self.click3)
        self.main_layout.addWidget(self.face_recognition_button)

    def click3(self):
        camera_reader.read()

    def click2(self):
        model_train.start()

    def click1(self):
        import face
        self.one = face.Ui_Form()
        self.one.show()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main = Ui_Form()
    main.show()
    sys.exit(app.exec_())