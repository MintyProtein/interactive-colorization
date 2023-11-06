import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from ..utils import *


class BaseColorizer:
    def colorize_target(self, lineart, color_current, target_mask):
        return NotImplementedError()


class CldmColorizer(BaseColorizer):
    def __init__(self, model, sampler, n_steps: int, resolution):
        raise NotImplementedError()
        #self.model = model
        #self.sampler = sampler
        #self.n_steps = n_steps
        #self.resolution = resolution
        
        return
    
    def colorize_target(self, lineart, color_current, target_mask):
        #TODO: CldmColorizer
        return