# modified from https://github.com/anhtuan85/Data-Augmentation-for-Object-Detection/blob/master/augmentation.ipynb

import PIL #version 1.2.0
from PIL import Image #version 6.1.0
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
import random


from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

class AdjustContrast:
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, img, target):
        """
        img (PIL Image or Tensor): Image to be adjusted.
        """
        _contrast_factor = ((random.random() + 1.0) / 2.0) * self.contrast_factor
        img = F.adjust_contrast(img, _contrast_factor)
        return img, target

class AdjustBrightness:
    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, img,target):
        """
        img (PIL Image or Tensor): Image to be adjusted.
        """
        _brightness_factor = ((random.random() + 1.0) / 2.0) * self.brightness_factor
        img = F.adjust_brightness(img, _brightness_factor)
        
        return img, target

def lighting_noise(image):
    '''
        color channel swap in image
        image: A PIL image
    '''
    new_image = image
  
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), 
             (1, 2, 0), (2, 0, 1), (2, 1, 0))
    swap = perms[random.randint(0, len(perms)- 1)]
    new_image = F.to_tensor(new_image)
    new_image = new_image[swap, :, :]
    new_image = F.to_pil_image(new_image)

    return new_image

class LightingNoise:
    def __init__(self) -> None:
        pass

    def __call__(self, img,target):
        img_noise  = lighting_noise(img)
        return img_noise, target


        
class RandomSelectMulti(object):
    """
    Randomly selects between transforms1 and transforms2,
    """
    def __init__(self, transformslist, p=-1):
        self.transformslist = transformslist
        self.p = p
        assert p == -1

    def __call__(self, img, target):
        if self.p == -1:
            return random.choice(self.transformslist)(img,target)


