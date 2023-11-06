import cv2
import numpy as np

class BaseRegionSelecter:
    def decide_next(self):
        return NotImplemented
    
class RandomRegionSelecter(BaseRegionSelecter):
    def __init__(self):
        return
    
    def decide_next(self, color, segmentation_map):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        next_region = np.zeros_like(gray)
        uncolored_labels = np.unique(segmentation_map[(segmentation_map > 0) & (gray < 255)])
        next_label = np.random.choice(uncolored_labels)
        next_region[segmentation_map==next_label] = 1
        return next_region