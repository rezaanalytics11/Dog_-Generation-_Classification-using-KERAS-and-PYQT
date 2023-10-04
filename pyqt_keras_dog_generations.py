from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import *
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QLabel, QFormLayout
from PyQt5.QtWidgets import (QApplication, QWidget,
  QPushButton, QVBoxLayout, QHBoxLayout,QGridLayout,QLineEdit)
import cv2
from tensorflow import keras
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image


Image_Width=224
Image_Height=224
Image_Size=(Image_Width,Image_Height)
Image_Channels=3
batch_size=15

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        hbox0 = QHBoxLayout()
        self.label0 = QLabel('Please enter the file name and press the key', self)
        hbox0.addWidget(self.label0)
        self.label0.setFont(QFont('Arial', 10))

        self.file_name = QLineEdit(self)
        hbox0.addWidget(self.file_name)

        hbox1 = QHBoxLayout()

        self.label1 = QLabel('', self)
        hbox1.addWidget(self.label1)
        self.label1.setFont(QFont('Arial', 20))

        self.label3 = QLabel('', self)
        hbox1.addWidget(self.label3)
        self.label3.setFont(QFont('Arial', 20))

        hbox2 = QHBoxLayout()
        self.label4 = QLabel('', self)
        hbox2.addWidget(self.label4)
        self.label3.setFont(QFont('Arial', 20))

        Button = QPushButton('Key')
        hbox0.addWidget(Button)
        Button.clicked.connect(self.addurl)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox0)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        self.setLayout(vbox)
        #self.setGeometry(400, 400, 300, 150)
        self.setWindowTitle('Dog Generation Classification Dialogue Box')
        self.show()

    def addurl(self):

        a=self.file_name.text()
        self.draw(a)

    def draw(self,file_name):
        model = keras.models.load_model(r'C:\Users\Ariya Rayaneh\Desktop\dog_model.h5')
        image = cv2.imread(rf'C:\Users\Ariya Rayaneh\Desktop\{file_name}.jpg')

        #img = img_to_array(image, data_format='channels_first')
        cv2.imwrite(r"C:\Users\Ariya Rayaneh\Desktop\dog_output.jpg", image)
        self.label1.setPixmap(QPixmap(r"C:\Users\Ariya Rayaneh\Desktop\dog_output.jpg"))

        results = {
            0: 'Chihuahua',
            1: 'Bedlington_terrier',
            2: 'Mexican_hairless'
        }

        image = cv2.resize(image, Image_Size)
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        image = image / 255.0
        pred = np.argmax(model.predict([image])[0])
        print(pred, results[pred])
        self.label3.setText(str(results[pred]))

if __name__ == '__main__':
 app = QApplication(sys.argv)
 ex = Example()
 sys.exit(app.exec_())