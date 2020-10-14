import cv2
import glob
import imutils
import os

def main():

    img_paths = glob.glob("downloads/*.jpg")
    digit_counter = {}

    # Create dataset directory
    os.makedirs("dataset", exist_ok=True)
    for i, img_path in enumerate(img_paths):
        print(f"[INFO] Processing image {i + 1}/{len(img_paths)}")
        try:

            image = cv2.imread(img_path)
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
            # Image segmentation by thresholding
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Find contours of digits
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

            for contour in contours:
                # Get bounding box for digit
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray[y-5:y+h+5, x-5:x+w+5]
                # Display image
                cv2.imshow("digit", imutils.resize(roi, width=28))
                key = cv2.waitKey(0)
                # Skip image
                if key == ord('`'):
                    print("[INFO] Ignore character")
                    continue
                # Save image to dataset
                key = chr(key).upper()
                save_path = f"dataset/{key}"
                os.makedirs(save_path, exist_ok=True)

                count = digit_counter.get(key, 1)
                cv2.imwrite(f"dataset/{key}/{count:06d}.png", roi)
                digit_counter[key] = count + 1

        except KeyboardInterrupt:
            print("[INFO] Leaving script ...")
            break

        except:
            print("[INFO] Skipping image ...")

if __name__ == '__main__':
    main()
