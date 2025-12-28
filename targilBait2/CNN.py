import cv2 as cv
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


def normalize(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Resize for CNN
    img = cv.resize(img, (150,100))

    # Normalize
    img = img.astype("float32") / 255
    return img

def load_and_split():
    X = []
    Y = []
    path = "rps-cv-images"

    for label, folder in enumerate(os.listdir(path)):
        folder_path = os.path.join(path, folder)

        for photo in os.listdir(folder_path):
            photo_path = os.path.join(folder_path, photo)
            img = normalize(photo_path)

            X.append(img)
            Y.append(label)  # convert class name → number

    X = np.array(X)
    Y = np.array(Y)

    # Correct output order: X_train, X_test, Y_train, Y_test
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def create_CNN_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,100, 3)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')   # Rock / Paper / Scissors
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_and_split()

    model = create_CNN_model()
    model.fit(X_train, Y_train, batch_size=32, epochs=30)

    loss, acc = model.evaluate(X_test, Y_test)
    print(f"Test accuracy: {acc:.4f}")
