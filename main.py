import numpy as np
import cv2
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Path to your dataset
train_dir = 'data/train'
validation_dir = 'data/validation'

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1.0/255)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Model definition using Functional API
inputs = Input(shape=(150, 150, 3))
x = Conv2D(100, (3,3), activation='relu')(inputs)
x = MaxPooling2D(2,2)(x)
x = Conv2D(100, (3,3), activation='relu')(x)
x = MaxPooling2D(2,2)(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(50, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Adding checkpoints to save the best model
checkpoint = ModelCheckpoint('model-{epoch:03d}.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


# Train model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# Load best model
model = load_model('model-010.keras')

# Real-time mask detection
cap = cv2.VideoCapture(0)
haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    faces = haarcascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (150, 150))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0) if label == 1 else (0, 0, 255), 2)
        cv2.putText(frame, 'Mask' if label == 1 else 'No Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Face Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
