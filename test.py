import argparse
import cv2
import glob
import imutils
import numpy as np
from core.utils.helpers import resize_with_pad
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to captcha files")
    parser.add_argument("-w", "--weights", default="weights.hdf5", help="Path to saved weights")
    args = parser.parse_args()

    # Load model
    model = load_model(args.weights)

    # Sample random images from file
    img_paths = glob.glob(f"{args.input}/*.jpg")
    img_paths = np.random.choice(img_paths, size=(10, ), replace=False)

    for img_path in img_paths:

        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

        # Initialize output
        output = cv2.merge([gray] * 3)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = gray[y-5:y+h+5, x-5:x+w+5]

            # Preprocess input
            roi = resize_with_pad(roi, 28, 28)
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0) / 255.

            pred = model.predict(roi)
            pred = pred.argmax(axis=1)[0] + 1

            cv2.rectangle(output, (x-2, y-2), (x+w+4, y+h+4), (0, 255, 0), 1)
            cv2.putText(output, str(pred), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("Output", output)
        cv2.waitKey(0)






if __name__ == '__main__':
    main()
