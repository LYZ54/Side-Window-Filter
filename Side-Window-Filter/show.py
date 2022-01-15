import cv2
import matplotlib.pyplot as plt


def bgr2rgb(img):
    b, g, r = cv2.split(img)
    img_rgb = cv2.merge([r, g, b])
    return img_rgb


def show(img):
    img = bgr2rgb(img)
    plt.imshow(img)
    plt.show()


def compare_show(img1, img2):
    img1 = bgr2rgb(img1)
    img2 = bgr2rgb(img2)
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()
