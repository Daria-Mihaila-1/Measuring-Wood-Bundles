import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
def plot_range(lower_color, higher_color):
    s_gradient = np.ones((500, 1), dtype=np.uint8)*np.linspace(lower_color[1], higher_color[1], 500, dtype=np.uint8)
    v_gradient = np.rot90(np.ones((500,1), dtype=np.uint8)*np.linspace(lower_color[1], higher_color[1], 500, dtype=np.uint8))
    h_array = np.arange(lower_color[0], higher_color[0]+1)

    for hue in h_array:
        h = hue*np.ones((500,500), dtype=np.uint8)
        hsv_color = cv2.merge((h, s_gradient, v_gradient))
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        cv2.imshow('', rgb_color)
        cv2.waitKey(250)

    cv2.destroyAllWindows()


def segment_img(image):
    matplotlib.use('TkAgg')
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    r = cv2.selectROI("select the area", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Display cropped image
    selected_image = image[r[1]:(r[1] + r[3]), r[0]:(r[0] + r[2])]

    rect = (int(r[0]), int(r[1]), r[2], r[3])
    cv2.imshow("Selected area", selected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Convert mask to displayable format (0 for background, 1 for foreground)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]

    cv2.imshow("Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Combine the original image with the overlay
    # Display the segmented image
    return image


def crop_img(image, left_margin, right_margin, top_margin, bottom_margin):
    height, width = image.shape[:2]

    x1 = left_margin
    x2 = width - right_margin
    y1 = top_margin
    y2 = height - bottom_margin

    cropped_image = image[y1:y2, x1:x2]

    return cropped_image


if __name__ == '__main__':
    img = cv2.imread("data/1.jpg")
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

    cv2.imshow("original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("image to segment", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    segment_img(img)