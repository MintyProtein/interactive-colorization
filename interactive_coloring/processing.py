import cv2
import numpy as np

def preprocess_lineart(lineart: np.ndarray):
    if lineart.ndim == 3:
        gray = cv2.cvtColor(lineart, cv2.COLOR_BGR2GRAY)
    else:
        gray = lineart.copy()

    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5).astype(np.uint8)

def preprocess_color():
    return NotImplementedError()

def postprocess():
    return NotImplementedError()