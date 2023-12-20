import cv2
import numpy as np

image_raw = cv2.imread("motherboard_image.JPEG", cv2.IMREAD_COLOR)

# Thresholoding
image_threshold = cv2.adaptiveThreshold(
    cv2.GaussianBlur(
        cv2.cvtColor(
            image_raw, 
            cv2.COLOR_BGR2GRAY), 
            (75, 75), 
            4
    ), 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 
    25, 
    5
)

# Edge Detection
corners_canny = cv2.Canny(image_threshold, 1, 1)
corners_dilated = cv2.dilate(corners_canny, None, iterations=7)

contours, _ = cv2.findContours(
    image=corners_dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
)
min_contour_area = 300
contours_filtered = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

image_countour = np.ones_like(image_raw) * 255
cv2.drawContours(image=image_countour, contours=contours, contourIdx=-1, color=(0, 0, 0))

# Masking
image_mask = np.zeros_like(image_raw)
cv2.drawContours(
    image=image_mask, contours=[max(contours_filtered, key=cv2.contourArea)],
    contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED
)
image_masked = cv2.bitwise_and(image_mask, image_raw)

cv2.imwrite("outputs/threshold.jpeg", image_threshold)
cv2.imwrite("outputs/countour.jpeg", image_countour)
cv2.imwrite("outputs/mask.jpeg", image_mask)
cv2.imwrite("outputs/final.jpeg", image_masked)