import argparse
import cv2
import glob
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import tqdm
from core.callbacks import TrainingMonitor
from core.nn.conv import LeNet
from core.utils.helpers import resize_with_pad
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Path to dataset")
    args = parser.parse_args()

    data = []
    labels = []
    for img_path in tqdm.tqdm(glob.glob(f"{args.dataset}/*/*.png")):
        # Get image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = resize_with_pad(image, 28, 28)
        image = img_to_array(image)
        data.append(image)
        # Get label
        label = img_path.split(os.path.sep)[-2]
        labels.append(label)

    # Standardize data
    data = np.array(data, dtype="float") / 255.

    # One hot encoding
    labels = np.array(labels)
    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    # Compile model
    model = LeNet.build(28, 28, 1, classes=9)
    optimizer = optimizers.SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    os.makedirs("outputs", exist_ok=True)
    fig_path = f"outputs/{os.getpid()}.png"
    json_path = f"outputs/{os.getpid()}.json"
    training_monitor = TrainingMonitor(fig_path, json_path=json_path)
    checkpoint = ModelCheckpoint("weights.hdf5", monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    callbacks = [training_monitor, checkpoint]

    # Train model
    H = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=15, callbacks=callbacks, verbose=1)

    model = load_model("weights.hdf5")
    preds = model.predict(X_test, batch_size=32)
    report = classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=label_binarizer.classes_)
    print(report)



if __name__ == '__main__':
    main()
