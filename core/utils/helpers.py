import cv2
import imutils

def resize_with_pad(image, height, width):

    img_height, img_width = image.shape[:2]

    # Resize image keeping aspect ratio
    if img_width > img_height:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)

    # Pad image
    img_height, img_width = image.shape[:2]
    pad_w = (width - img_width) // 2
    pad_h = (height - img_height) // 2
    image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)

    # Resize image to ensure correct dimension
    image = cv2.resize(image, (width, height))
    return image
