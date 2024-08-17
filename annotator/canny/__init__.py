"""
Applies the Canny Edge Detection Algorithm (CEDA) to the input image.

Args:
    img (numpy.ndarray): The input image.
    low_threshold (int): The lower threshold for hysteresis.
    high_threshold (int): The upper threshold for hysteresis.

Returns:
    numpy.ndarray: The edge-detected image.
"""
# Note: For more detail regarding CEDA refer to https://en.wikipedia.org/wiki/Canny_edge_detector#:~:text=The%20Canny%20edge%20detector%20is,explaining%20why%20the%20technique%20works.

import cv2


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)
