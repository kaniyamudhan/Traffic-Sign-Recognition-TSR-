#by kaniyamudhan.Y
from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pyttsx3
import matplotlib.pyplot as plt
from PIL import Image
import os

data = []
labels = []
classes = 43
cur_path = os.getcwd()  # To get current directory

classs = {1: "Speed limit (20km/h)",
          2: "Speed limit (30km/h)",
          3: "Speed limit (50km/h)",
          4: "Speed limit (60km/h)",
          5: "Speed limit (70km/h)",
          6: "Speed limit (80km/h)",
          7: "End of speed limit (80km/h)",
          8: "Speed limit (100km/h)",
          9: "Speed limit (120km/h)",
          10: "No passing",
          11: "No passing veh over 3.5 tons",
          12: "Right-of-way at intersection",
          13: "Priority road",
          14: "Yield",
          15: "Stop",
          16: "No vehicles",
          17: "Veh > 3.5 tons prohibited",
          18: "No entry",
          19: "General caution",
          20: "Dangerous curve left",
          21: "Dangerous curve right",
          22: "Double curve",
          23: "Bumpy road",
          24: "Slippery road",
          25: "Road narrows on the right",
          26: "Road work",
          27: "Traffic signals",
          28: "Pedestrians",
          29: "Children crossing",
          30: "Bicycles crossing",
          31: "Beware of ice/snow",
          32: "Wild animals crossing",
          33: "End speed + passing limits",
          34: "Turn right ahead",
          35: "Turn left ahead",
          36: "Ahead only",
          37: "Go straight or right",
          38: "Go straight or left",
          39: "Keep right",
          40: "Keep left",
          41: "Roundabout mandatory",
          42: "End of no passing",
          43: "End no passing veh > 3.5 tons"}

# Retrieving the images and their labels
print("Obtaining Images & its Labels..............")
for i in range(classes):
    path = os.path.join(cur_path, 'dataset/train/', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '\\' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
            print("{0} Loaded".format(a))
        except:
            print("Error loading image")
print("Dataset Loaded")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

#by kaniyamudhan.Y
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(160, 370, 151, 51))
        self.BrowseImage.setObjectName("BrowseImage")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 621, 20))
        font = QtGui.QFont()
        font.setFamily("Courier New")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))
        self.Classify.setObjectName("Classify")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(430, 370, 111, 16))
        self.label.setObjectName("label")
        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(400, 450, 151, 51))
        self.Training.setObjectName("Training")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(400, 390, 211, 51))
        self.textEdit.setObjectName("textEdit")
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
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

    # You can adjust the speaking speed (optional)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BrowseImage.setText(_translate("MainWindow", "Browse Image"))
        self.label_2.setText(_translate("MainWindow", "           ROAD SIGN RECOGNITION"))
        self.Classify.setText(_translate("MainWindow", "Classify"))
        self.label.setText(_translate("MainWindow", "Recognized Class"))
        self.Training.setText(_translate("MainWindow", "Training"))

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "",
                                                            "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)")  # Ask for file
        if fileName:  # If the user gives a file
            print(fileName)
            self.file = fileName
            pixmap = QtGui.QPixmap(fileName)  # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.imageLbl.setPixmap(pixmap)  # Set the pixmap onto the label
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center
#by kaniyamudhan.Y
    def classifyFunction(self):
        model = load_model('my_model.h5')
        print("Loaded model from disk");
        path2 = self.file
        print(path2)
        test_image = Image.open(path2)
        test_image = test_image.resize((30, 30))
        test_image = np.expand_dims(test_image, axis=0)
        test_image = np.array(test_image)

        result = model.predict(test_image).argmax(axis=-1)[0]

        sign = classs[result + 1]
        print(sign)
        self.textEdit.setText(sign)
        self.speak(sign)  # Speak the recognized sign

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def trainingFunction(self):
        self.textEdit.setText("Training under process...")
        self.speak("Training completed. Model saved.")
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(43, activation='softmax'))
        print("Initialized model")

        # Compilation of the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))
        model.save("my_model.h5")
#by kaniyamudhan.Y
        plt.figure(0)
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig('Accuracy.png')

        plt.figure(1)
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('Loss.png')
        self.textEdit.setText("Saved Model & Graph to disk")

#by kaniyamudhan.Y
if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

