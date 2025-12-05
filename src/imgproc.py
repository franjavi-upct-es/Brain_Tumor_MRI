import cv2
import numpy as np


def crop_brain_contour(image_path_or_array, plot=False):
    """
    Loads an image, detects the largest contour (the brain), and removes excessive black background.

    Args:
        image_path_or_array: Path to file (str) or numpy array (BGR).
        plot: (bool) If True, reserved for visual debug (currently unused).

    Returns:
        numpy.ndarray: The cropped image. Returns original if detections fails.
    """
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
    else:
        image = image_path_or_array

    if image is None:
        return None

    # Convert to grayscale and blur to remove noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to separate brain (bright object) from background
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

    # Morphological operations to clean imperfections
    thres = cv2.erode(thresh, None, iterations=2)
    thres = cv2.dilate(thres, None, iterations=2)

    # Find contours
    cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if not cnts:
        return image  # Fallback if nothing detected

    # Take the largest contour
    c = max(cnts, key=cv2.contourArea)

    # Get extreme coordinates
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBottom = tuple(c[c[:, :, 1].argmax()][0])

    # Add small padding if posible
    pad = 0
    h, w = image.shape[:2]
    y1 = max(0, extTop[1] - pad)
    y2 = min(h, extBottom[1] + pad)
    x1 = max(0, extLeft[0] - pad)
    x2 = min(w, extRight[0] + pad)

    # Crop
    new_image = image[y1:y2, x1:x2]

    return new_image
