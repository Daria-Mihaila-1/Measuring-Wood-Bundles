import os.path

import cv2
import numpy as np

from utility import segment_img, crop_img

def normalize_hsv_values(hue, saturation, value):
    # Convert hue from degrees to OpenCV's range (0-179)
    hue_for_cv2 = int(hue / 2)

    # Normalize saturation and value to the range of 0-255
    saturation_for_cv2 = int(saturation * 255 / 100)
    value_for_cv2 = int(value * 255 / 100)
    return [hue_for_cv2, saturation_for_cv2, value_for_cv2]


def create_hsv_color_mask(rgb_lighter_color, hue_range, saturation, value):
    hsv_color = cv2.cvtColor(rgb_lighter_color, cv2.COLOR_RGB2HSV)[0][0]
    hue = hsv_color[0]
    print(hue)
    print(hsv_color)
    upper_color = np.array([hue + hue_range,  saturation[1], value[1]])
    print("uppercolor", upper_color)
    lower_color = np.array([hue - hue_range, saturation[0], value[0]])
    print("lowercolor", lower_color)

    if lower_color[0] < 0:
        # the desired hue goes from negative to positive
        # cv.inRange has no logic that would allow you to express that in a single range.
        lower_upper_color = np.array([0, saturation[0], value[0]])
        mask1 = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_upper_color, upper_color)

        upper_lower_color = np.array([179, saturation[1], value[1]])
        new_lower_color = np.array([179 + lower_color[0] - 20, lower_color[1], lower_color[2]])
        mask2 = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), new_lower_color, upper_lower_color)
        return mask1 + mask2

    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_color, upper_color)
    return mask


if __name__ == "__main__":

    img_path = "./data/1.jpg"
    img = cv2.imread(img_path)
    width, height = img.shape[:2]

    img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)

    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = segment_img(img)

    rgb_red = np.uint8([[[214, 57, 59]]])
    rgb_green = np.uint8([[[19, 111, 87]]])
    rgb_blue = np.uint8([[[28, 45, 91]]])
    rgb_cyan = np.uint8([[[11, 141, 238]]])

    red_saturation_limits = [40, 255]
    green_saturation_limits = [60, 255]
    blue_saturation_limits = [100, 255]
    cyan_saturation_limits = [100, 255]

    red_value_limits = [100, 255]
    green_value_limits = [70, 255]
    blue_value_limits = [10, 160]
    cyan_value_limits = [100, 255]

    red_mask = create_hsv_color_mask(rgb_red, 3, red_saturation_limits, red_value_limits)
    green_mask = create_hsv_color_mask(rgb_green, 15, green_saturation_limits, green_value_limits)
    blue_mask = create_hsv_color_mask(rgb_blue, 12, blue_saturation_limits, blue_value_limits)
    cyan_mask = create_hsv_color_mask(rgb_cyan, 13, cyan_saturation_limits, cyan_value_limits)

    combined_mask = cv2.bitwise_or(green_mask, red_mask)
    combined_mask = cv2.bitwise_or(combined_mask, blue_mask)
    combined_mask = cv2.bitwise_or(combined_mask, cyan_mask)

    # Apply the mask to the original image to extract pixels of the specified color
    result = cv2.bitwise_and(img, img, mask=combined_mask)

    cv2.imshow("original image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("Resulting mask", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    combined_mask = cv2.bilateralFilter(combined_mask,  9, 75, 75)  # Adjust kernel size as neede
    cv2.imshow("Resulting mask after gaussian blurr", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    relevant_contours = [contour for contour in contours if 100 < cv2.contourArea(contour) < 900]
    for contour in relevant_contours:
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 1)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(len(relevant_contours))
    for contour in relevant_contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 1)
            cv2.circle(img, (cx, cy), 2, (255, 255, 0), -1)
            cv2.putText(img, "centroid", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.imshow("Centroids found", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
