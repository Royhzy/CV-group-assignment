from tkinter import messagebox
from PyQt5.QtWidgets import QWidget, QApplication, QTextBrowser, QLineEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QVBoxLayout
from PyQt5 import QtCore, QtGui
import sys
from PyQt5.QtCore import QTimer
import pymysql

class Ui_Form(QWidget):
    def __init__(self, parent=None):
        super(Ui_Form, self).__init__(parent)

        self.setWindowTitle("Login")
        self.setStyleSheet("background-color: #f0f4f8;")  # Light background color
        self.setFixedSize(600, 310)

        # Create layouts
        self.main_layout = QVBoxLayout()
        self.form_layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()

        # Welcome message
        self.textBrowser = QTextBrowser(self)
        self.textBrowser.setHtml("<h1>Welcome</h1>")
        self.textBrowser.setStyleSheet("color: #2C3E50; font-family: 'Arial'; font-weight: bold; text-align: center;")
        self.form_layout.addWidget(self.textBrowser)

        # Account label and input
        self.label = QLabel("Account:", self)
        self.label.setStyleSheet("font-size: 18px; color: #34495e;")
        self.form_layout.addWidget(self.label)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setPlaceholderText("Enter Account")
        self.lineEdit.setStyleSheet("border: 1px solid #bdc3c7; border-radius: 15px; padding: 10px; font-size: 16px;")
        self.form_layout.addWidget(self.lineEdit)

        # Password label and input
        self.label_2 = QLabel("Password:", self)
        self.label_2.setStyleSheet("font-size: 18px; color: #34495e;")
        self.form_layout.addWidget(self.label_2)

        self.lineEdit_2 = QLineEdit(self)
        self.lineEdit_2.setPlaceholderText("Enter Password")
        self.lineEdit_2.setEchoMode(QLineEdit.Password)  # Mask password input
        self.lineEdit_2.setStyleSheet("border: 1px solid #bdc3c7; border-radius: 15px; padding: 10px; font-size: 16px;")
        self.form_layout.addWidget(self.lineEdit_2)

        # Button
        self.pushButton = QPushButton("Login", self)
        self.pushButton.setStyleSheet("background-color: #3498db; color: white; border-radius: 15px; font-size: 18px; padding: 12px;")
        self.pushButton.setCursor(QtCore.Qt.PointingHandCursor)  # Change cursor to pointer
        self.pushButton.clicked.connect(self.login)
        self.button_layout.addWidget(self.pushButton)

        # Add form layout and button layout to main layout
        self.main_layout.addLayout(self.form_layout)
        self.main_layout.addLayout(self.button_layout)

        self.setLayout(self.main_layout)  # Set main layout for the widget

    def login(self):
        if len(self.lineEdit.text()) == 0:
            messagebox.showinfo("warning", "Please enter account!")
            #self.textBrowser.setText("Please enter account")
            return
        elif len(self.lineEdit_2.text()) == 0:
            messagebox.showinfo("warning", "Please enter password!")
            #self.textBrowser.setText("Please enter password")
            return
        QTimer.singleShot(2000, self.check_login)

    def check_login(self):
        con = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='root',
            charset='utf8',
            database='face'
        )
        sql = "select password from login where name = '%s'" % self.lineEdit.text()
        cursor = con.cursor()
        cursor.execute(sql)
        res = cursor.fetchone()
        if res:
            result = ''.join(res)
            if result == self.lineEdit_2.text():
                self.textBrowser.setText("success to login")
                QTimer.singleShot(2000, self.close)
                QTimer.singleShot(2000, self.success)
                messagebox.showinfo("success", "success to login!")
                #print("success to login")
            else:
                messagebox.showinfo("warning", "wrong password or account,please re-enter!")
                #self.textBrowser.setText("wrong password or account,please re-enter")
        else:
            messagebox.showinfo("warning", "Account does not exist!")
            #self.textBrowser.setText("Account does not exist")
    def success(self):
        import main
        self.one = main.Ui_Form()
        self.one.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Ui_Form()
    main.show()
    sys.exit(app.exec_())