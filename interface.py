# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Rabbit\Documents\大四上\影像處理\作業\Hw2\hw2-5\interface.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(197, 283)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.show_model_Button = QtWidgets.QPushButton(self.centralwidget)
        self.show_model_Button.setGeometry(QtCore.QRect(30, 40, 141, 21))
        self.show_model_Button.setObjectName("show_model_Button")
        self.show_tensor_Button = QtWidgets.QPushButton(self.centralwidget)
        self.show_tensor_Button.setGeometry(QtCore.QRect(30, 80, 141, 21))
        self.show_tensor_Button.setObjectName("show_tensor_Button")
        self.test_Button = QtWidgets.QPushButton(self.centralwidget)
        self.test_Button.setGeometry(QtCore.QRect(30, 120, 141, 21))
        self.test_Button.setObjectName("test_Button")
        self.data_augmantation_Button = QtWidgets.QPushButton(self.centralwidget)
        self.data_augmantation_Button.setGeometry(QtCore.QRect(30, 200, 141, 21))
        self.data_augmantation_Button.setObjectName("data_augmantation_Button")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(30, 159, 141, 21))
        self.lineEdit.setObjectName("lineEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.show_model_Button.setText(_translate("MainWindow", "1.Show Model Structure"))
        self.show_tensor_Button.setText(_translate("MainWindow", "2.Show TensorBoard"))
        self.test_Button.setText(_translate("MainWindow", "3.Test"))
        self.data_augmantation_Button.setText(_translate("MainWindow", "4.Data Augmantation"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())