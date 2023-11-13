import cv2
import numpy as np

class BaseRegionSelecter:
    def decide_next(self):
        return NotImplementedError()
    def get_color_mask(self):
        return NotImplementedError()
    
class RandomRegionSelecter(BaseRegionSelecter):
    def __init__(self):
        return
    
    def decide_next(self, color, segmentation_map, n=1):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        target_mask = np.zeros_like(gray)
        uncolored_labels = np.unique(segmentation_map[(segmentation_map > 0) & (gray < 255)])
        
        n_next = min(len(uncolored_labels), n)
        next_labels = np.random.choice(uncolored_labels, size=n_next, replace=False)
        for label in next_labels: 
            target_mask[segmentation_map==label] = 1
            
        return target_mask
    
    def get_color_mask(self, color):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        mask = ((color[:, :, 0] == color[:,:,1]) & (color[:,:,1]== color[:,:,2])).astype(np.uint8)
        mask[gray==0] = 0
        return mask