from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import os
import cv2
import numpy as np

model_path = 'model-010.keras'

def create_model():
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
    return model

if os.path.exists(model_path):
    print("Loading the existing model.")
    #model = load_model(model_path)
    model = load_model('model-010.keras')
else:
    print("No existing model found. Training a new one.")
    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    checkpoint = ModelCheckpoint('model-{epoch:03d}.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    history = model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=[checkpoint])

    model = load_model('model-010.keras')

haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rect_size = 4
results = {0: 'without mask', 1: 'with mask'}
GR_dict = {0: (0,0,255), 1: (0,255,0)}

while True:
    rval, im = cap.read()
    im = cv2.flip(im, 1, 1) 

    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]
        face_img = im[y:y+h, x:x+w]
        rerect_sized = cv2.resize(face_img, (150,150))
        normalized = rerect_sized / 255.0
        reshaped = np.reshape(normalized, (1,150,150,3))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(im, (x, y), (x+w, y+h), GR_dict[label], 2)
        cv2.rectangle(im, (x, y-40), (x+w, y), GR_dict[label], -1)
        cv2.putText(im, results[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()
