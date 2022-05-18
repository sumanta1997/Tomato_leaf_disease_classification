

from PyQt5 import QtCore, QtGui, QtWidgets
import time
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.preprocessing import image
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Dropout


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1393, 940)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(900, 190, 241, 71))
        self.BrowseImage.setObjectName("BrowseImage")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(90, 160, 751, 411))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 1121, 71))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(900, 290, 241, 71))
        self.Classify.setObjectName("Classify")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 590, 271, 91))
        self.label.setObjectName("label")
        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(890, 410, 261, 71))
        self.Training.setObjectName("Training")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(90, 680, 631, 81))
        self.textEdit.setObjectName("textEdit")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(890, 590, 256, 231))
        self.listWidget.setObjectName("listWidget")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(900, 520, 231, 51))
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1393, 43))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.BrowseImage.clicked.connect(self.loadImage)

        self.Classify.clicked.connect(self.classifyFunction)

        self.Training.clicked.connect(self.trainingFunction)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "Sumanta\'s Tomato Leaf Disease Classifier"))
        self.BrowseImage.setText(_translate("MainWindow", "OPEN"))
        self.label_2.setText(_translate(
            "MainWindow", "TOMATO LEAF DISEASE PREDICTION USING AI"))
        self.Classify.setText(_translate("MainWindow", "CLASSIFY"))
        self.label.setText(_translate("MainWindow", "RECOGNIZED CLASS"))
        self.Training.setText(_translate("MainWindow", "TRAINING"))
        self.label_3.setText(_translate("MainWindow", "RESULT LOGS"))

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "",
                                                            "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)")  # Ask for file
        if fileName:  # If the user gives a file
            print(fileName)
            self.file = fileName
            # Setup pixmap with the provided image
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(
            ), QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.imageLbl.setPixmap(pixmap)  # Set the pixmap onto the label
            # Align the label to center
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)
            self.textEdit.setText(" ")

    def classifyFunction(self):

        loaded_model = tf.keras.models.load_model('leaf_model.h5')

        print("Loaded model from disk")
        label = ['Bacterial Spot',
                 'Early Blight',
                 'Late Blight',
                 'Leaf Mold',
                 'Septoria Leaf Spot',
                 'Spider Mites',
                 'Target Spot',
                 'Yellow Leaf Curl Virus',
                 'Healthy',
                 'Mosaic Virus']

        path2 = self.file
        print(path2)
        test_image = image.load_img(path2, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = loaded_model.predict(test_image)

        fresult = np.max(result)
        label2 = label[result.argmax()]
        print(label2)
        self.textEdit.setText(label2)

    def trainingFunction(self):
        train_dir = "Dataset/Training Set"
        val_dir = "Dataset/Validation Set"
        test_dir = "Dataset/Test Set"

        train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                         image_size=(
                                                                             224, 224),
                                                                         label_mode="categorical",
                                                                         batch_size=32)
        val_data = tf.keras.preprocessing.image_dataset_from_directory(directory=val_dir,
                                                                       image_size=(
                                                                           224, 224),
                                                                       label_mode="categorical",
                                                                       batch_size=32)

        # 1.Create baseline model with tf.keras.applications
        base_model = tf.keras.applications.EfficientNetV2B0(include_top=False)

        # To begin fine-tuning, let's start by setting the last 10 layers of our base_model.trainable = True
        base_model.trainable = False

        # 3.Create inputs to the model
        inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

        # 5.pass the inputs to the baseline model
        x = base_model(inputs, training=False)
        print(
            f"Shape of after passing inputs to the baseline model is {x.shape}")

        # 6.Average pool the outputs of the base model
        x = tf.keras.layers.GlobalAveragePooling2D(
            name="global_average_pooling_layer")(x)
        print(f"Shape after passing GlobalAveragePooling layer is {x.shape}")

        # 7.Ouput layer
        outputs = tf.keras.layers.Dense(
            10, activation="softmax", name="output_layer")(x)

        # 8.Combine the model
        model_0 = tf.keras.Model(inputs, outputs)

        # 9.Compile the model
        model_0.compile(loss="categorical_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(
                            learning_rate=0.005),
                        metrics=['accuracy'])

        # 11.Fit the model
        history_0 = model_0.fit(train_data,
                                steps_per_epoch=len(train_data),
                                epochs=40,
                                validation_data=val_data,
                                validation_steps=len(val_data)

                                )
        model_0.save('leaf_model.h5')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
