import numpy as np
from .processing import *
from models.regionSelecter import *
from models.colorizer import *
from models.segmenter import *

class InteractiveColorNet:
    def __init__(self, colorizer: BaseColorizer, segmenter: BaseSegmenter, region_selecter: BaseRegionSelecter):        
        self.lineart_segmenter = segmenter
        self.colorizer = colorizer
        self.region_selecter = region_selecter
        
        self._lineart = None
        self._num_regions = None
        self._segmentation_map = None
        return

    def set_lineart(self, img) -> None:
        self._lineart = preprocess_lineart(img)
        self.num_regions, self._segmentation_map = self.lineart_segmenter(self.lineart)
        return
    
    def __call__(self, color):
        if self._lineart is None or self._segmentation_map is None:
            raise AttributeError()
        
        color_current = preprocess_color(color)
        
        # Decide the next region to colorize
        target_mask = self.region_selecter.decide_next(color_current, self._segmentation_map)
        
        # Color the target region
        color_output = self.colorizer.colorize_target(self.lineart, color_current, target_mask)
        
        color_output = postprocess(color_output)
        
        return color_output
        
        
        
        
