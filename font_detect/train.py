import os
import json
import random
import numpy as np
from imutils import paths
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((105, 105))
    return img_to_array(img)

def load_dataset(dataset_path):
    data = []
    labels = []
    label_map = {}
    label_id = 0

    for label_name in sorted(os.listdir(dataset_path)):
        label_folder = os.path.join(dataset_path, label_name)
        if not os.path.isdir(label_folder):
            continue

        label_map[label_id] = label_name
        image_paths = list(paths.list_images(label_folder))

        for image_path in image_paths:
            try:
                img_array = preprocess_image(image_path)
                data.append(img_array)
                labels.append(label_id)
            except Exception as e:
                print(f"[WARNING] Skipping {image_path}: {e}")
        label_id += 1

    return np.array(data, dtype="float") / 255.0, np.array(labels), label_map

def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(105, 105, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def main():
    dataset_path = "font_dataset"
    model_output = "font_model.h5"
    label_map_output = "label_map.json"

    print("[INFO] Loading dataset...")
    data, labels, label_map = load_dataset(dataset_path)
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
    trainY = to_categorical(trainY, num_classes=len(label_map))
    testY = to_categorical(testY, num_classes=len(label_map))

    print("[INFO] Building model...")
    model = build_model(num_classes=len(label_map))
    opt = optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[INFO] Training model...")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, verbose=1),
        ModelCheckpoint(model_output, save_best_only=True, monitor="val_loss", verbose=1)
    ]
    model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=50, callbacks=callbacks, verbose=1)

    print("[INFO] Saving label map...")
    with open(label_map_output, "w") as f:
        json.dump(label_map, f)

    print("[INFO] Done. Model and label map saved.")

if __name__ == "__main__":
    main()
