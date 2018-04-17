import numpy as np
import cv2


def resize_image():
    print("Resizing Image")
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE) # Load the image.
    resized_image = cv2.resize(img, (28, 28)) # Resizing the image into 28x28 matrix

    cv2.imshow('image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Resizing Completed.")
    return resized_image.flatten()


if __name__ == "__main__":
    resize_image()
