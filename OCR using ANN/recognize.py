import numpy as np
import cv2


def resize_image():
    print("Resizing Image")
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE) # Load the image.
    norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    resized_image = cv2.resize(norm_image, (28, 28)) # Resizing the image into 28x28 matrix

    # cv2.imshow('image', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("Resizing Completed.")
    return resized_image.flatten()


if __name__ == "__main__":
    resize_image()
