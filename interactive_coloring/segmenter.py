import cv2

class BaseSegmenter:
    """
    Base [Segmenter] to get segmentation map from line art.
    """
    def __call__(self):
        return NotImplementedError()
    

class SimpleSegmenter(BaseSegmenter):
    """
    A simple [Segmenter] class using 'cv2.connectedComponents()'
    """
    def __call__(self, lineart):
        return cv2.connectedComponents(lineart)
