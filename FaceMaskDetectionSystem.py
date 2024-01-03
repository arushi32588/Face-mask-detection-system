from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *

model = Sequential([
    Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

checkpoint = ModelCheckpoint('model2-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

history = model.fit_generator(train_generator, epochs=10, validation_data=validation_generator, callbacks=[checkpoint])

model = load_model("./model-010.h5")
results = {0: 'without mask', 1: 'mask'}
GR_dict = {0: (0,0,255), 1: (0,255,0)}
rect_size = 4
cap = cv2.VideoCapture(0)
haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    (rval, im) = cap.read()
    im = cv2.flip(im, 1, 1)

    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]

        face_img = im[y:y+h, x:x+w]
        rerect_sized = cv2.resize(face_img, (150,150))
        normalized = rerect_sized / 255.0
        reshaped = np.reshape(normalized, (1,150,150,3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(im, (x,y), (x+w,y+h), GR_dict[label], 2)
        cv2.rectangle(im, (x,y-40), (x+w,y), GR_dict[label], -1)
        cv2.putText(im, results[label], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

a = input("Do you want to know why face mask detection is important? Answer in yes/no")
if a == "yes":
    class MainWindow(QMainWindow):
        def __init__(self):
            super(MainWindow, self).__init__()
            self.browser = QWebEngineView()
            self.browser.setUrl(QUrl('http://google.com'))
            self.setCentralWidget(self.browser)
            self.showMaximized()

            navbar = QToolBar()
            self.addToolBar(navbar)

            back_btn = QAction('Back', self)
            back_btn.triggered.connect(self.browser.back)
            navbar.addAction(back_btn)

            forward_btn = QAction('Forward', self)
            forward_btn.triggered.connect(self.browser.forward)
            navbar.addAction(forward_btn)

            reload_btn = QAction('Reload', self)
            reload_btn.triggered.connect(self.browser.reload)
            navbar.addAction(reload_btn)

            home_btn = QAction('Home', self)
            home_btn.triggered.connect(self.navigate_home)
            navbar.addAction(home_btn)

            self.url_bar = QLineEdit()
            self.url_bar.returnPressed.connect(self.navigate_to_url)
            navbar.addWidget(self.url_bar)

            self.browser.urlChanged.connect(self.update_url)

        def navigate_home(self):
            self.browser.setUrl(QUrl('http://programming-hero.com'))

        def navigate_to_url(self):
            url = self.url_bar.text()
            self.browser.setUrl(QUrl(url))

        def update_url(self, q):
            self.url_bar.setText(q.toString())

    app = QApplication([])
    QApplication.setApplicationName('My Cool Browser')
    window = MainWindow()
    app.exec_()
else:
    print("you're good to go")
