from PyQt5 import QtWidgets, QtGui, QtCore
import sys
from interface import Ui_MainWindow
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras
import cv2

class MainWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self,parent = None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.show_model_Button.clicked.connect(self.show_model)
        self.show_tensor_Button.clicked.connect(self.show_tensor)
        
        restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(244,300,3))
        self.model = Sequential()
        self.model.add(restnet)
        self.model.add(Dense(512, activation='relu', input_dim=(244,300,3)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                    optimizer=tensorflow.keras.optimizers.RMSprop(lr=2e-5),
                    metrics=['accuracy'])



    def show_model(self):
        self.model.summary()

    def show_tensor(self):
        fig= cv2.imread('./tensorboard.jpg')
        fig=cv2.resize(fig,(1200,500))
        cv2.imshow('TensorBoard',fig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__=='__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())