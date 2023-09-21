# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate
import math
import cv2


def hflip(image,image_all,target):
    flipped_image = F.hflip(image)
    flipped_image_all = F.hflip(image_all)

    w, h = image.size
    w_ori ,h_ori = image_all.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "small_sign_in_all" in target:
        small_sign_in_all = target["small_sign_in_all"]
        small_sign_in_all = small_sign_in_all[[2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([1, 0,1, 0])
        target["small_sign_in_all"] = small_sign_in_all
    

    return flipped_image, flipped_image_all,target


def resize(image,image_all, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        scales = [384,480,576,672,768,864,960]
        if w < h:
            ow = size
            if size  not in scales:
                oh = int(size * h / w)
            else:
                oh = int(round((size * h / w)/32)*32)
        else:
            oh = size
            if  size not in scales:
                ow = int(size * w / h)
            else:
                ow = int(round(size * w / h/32)*32)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    # print(size)
    rescaled_image = F.resize(image, size)

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    img_all_size = tuple(math.ceil(float(s) * float(s_orig)) for s, s_orig in zip(image_all.size, ratios))
    rescaled_image_all = F.resize(image_all, (img_all_size[1],img_all_size[0]))

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes
    
    if "small_sign" in target:
        small_sign = target["small_sign"]
        scaled_small_sign = small_sign * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["small_sign"] = scaled_small_sign
    
  

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image,rescaled_image_all,target



class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img,img_all, target):
        if random.random() < self.p:
            return hflip(img,img_all, target)
        return img,img_all, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img,img_all, target=None):
        size = random.choice(self.sizes)
        return resize(img, img_all,target, size, self.max_size)



class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img,img_all, target):
        if random.random() < self.p:
            return self.transforms1(img,img_all, target)
        return self.transforms2(img,img_all, target)


class ToTensor(object):
    def __call__(self, img,img_all, target):
        return F.to_tensor(img),F.to_tensor(img_all), target





class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image,image_all, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        image_all = F.normalize(image_all, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        h_ori, w_ori = image_all.shape[-2:]

        feature_h = h/32
        feature_w = w/32
        outpaint_h = round(feature_h*0.2)*32
        outpaint_w = round(feature_w*0.2)*32
        if "boxes" in target:
            boxes = target["boxes"]
            boxes[:,0]= boxes[:,0] + outpaint_w
            boxes[:,1]= boxes[:,1] + outpaint_h
            boxes[:,2]= boxes[:,2] + outpaint_w
            boxes[:,3]= boxes[:,3] + outpaint_h
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w+2*outpaint_w, h+2*outpaint_h, w+2*outpaint_w, h+2*outpaint_h], dtype=torch.float32)
            target["boxes"] = boxes
        
        if "orig_size" in target:
            orig_size = target["orig_size"] 
            scale_w = (2*outpaint_w+w)/w
            scale_h = (2*outpaint_h+h)/h
            orig_size[0] = orig_size[0]*scale_h
            orig_size[1] = orig_size[1]*scale_w
            target["orig_size"] = orig_size

        if "small_sign" in target:
            small_sign = target["small_sign"]
            small_sign[0]= small_sign[0] - outpaint_w
            small_sign[1]= small_sign[1] - outpaint_h
            small_sign[2]= small_sign[2] + outpaint_w
            small_sign[3]= small_sign[3] + outpaint_h
            small_sign = small_sign / torch.tensor([w+2*outpaint_w, h+2*outpaint_h, w+2*outpaint_w, h+2*outpaint_h], dtype=torch.float32)
            target["small_sign"] = small_sign
        
        #get ori_pic now crood
        #以原图为基础的小图坐标
        x_s_ori_crop = int(torch.round(target["small_sign_in_all"][0]*w_ori))
        y_s_ori_crop = int(torch.round(target["small_sign_in_all"][1]*h_ori))
        x_e_ori_crop = int(torch.round(target["small_sign_in_all"][2]*w_ori))
        y_e_ori_crop = int(torch.round(target["small_sign_in_all"][3]*h_ori))
        #应该正好是32的倍数 也是小图的大小
        
        x_e_ori_crop = x_s_ori_crop + round((x_e_ori_crop-x_s_ori_crop)/32)*32
        y_e_ori_crop = y_s_ori_crop + round((y_e_ori_crop-y_s_ori_crop)/32)*32

        
        x_left_int = x_s_ori_crop//32
        y_top_int = y_s_ori_crop//32
        x_right_int = x_left_int+ (x_e_ori_crop - x_s_ori_crop)//32
        y_bottom_int = y_top_int+ (y_e_ori_crop - y_s_ori_crop)//32

     
       

        target['outpainting'] = torch.tensor([int(x_left_int),int(y_top_int),int(x_right_int),int(y_bottom_int)])

        all_x_start = x_s_ori_crop - x_left_int*32
        all_y_start = y_s_ori_crop - y_top_int*32
        all_x_end = x_e_ori_crop +(w_ori -x_e_ori_crop)//32 *32
        all_y_end = y_e_ori_crop +(h_ori -y_e_ori_crop)//32 *32

        # name = str(target['image_id'][0].numpy())
        # save_dir = '%s/%s.jpg' % ('./test/pic', name+'_small_sign_in_all')
        # image_input = image_all[:,int(y_s_ori_crop):int(y_e_ori_crop),int(x_s_ori_crop):int(x_e_ori_crop)].permute(1, 2, 0).numpy()* 255
        # image_input = image_input[:, :, [2, 1, 0]]
        # cv2.imwrite(save_dir, image_input)

        # save_dir = '%s/%s.jpg' % ('./test/pic', name+'_img_all_real')
        # image_input = image_all.permute(1, 2, 0).numpy()* 255
        # image_input = image_input[:, :, [2, 1, 0]]
        # cv2.imwrite(save_dir, image_input)

        image_all = image_all[:,int(all_y_start):int(all_y_end),int(all_x_start):int(all_x_end)] 


        # save_dir = '%s/%s.jpg' % ('./test/pic', name+'_img_all')
        # image_input = image_all.permute(1, 2, 0).numpy()* 255
        # image_input = image_input[:, :, [2, 1, 0]]
        # cv2.imwrite(save_dir, image_input)

        # save_dir = '%s/%s.jpg' % ('./test/pic', name+'_img')
        # image_input = image.permute(1, 2, 0).numpy()* 255
        # image_input = image_input[:, :, [2, 1, 0]]
        # cv2.imwrite(save_dir, image_input)

        # save_dir = '%s/%s.jpg' % ('./test/pic', name+'_img_all_crop')
        # x_in_s,y_in_s,x_in_e,y_in_e= target['outpainting'][:]
        # image_input = image_all[:,y_in_s*32:y_in_e*32,x_in_s*32:x_in_e*32].permute(1, 2, 0).numpy()* 255
        # image_input = image_input[:, :, [2, 1, 0]]
        # cv2.imwrite(save_dir, image_input)

        
        return image,image_all,target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image,image_all, target):
        for t in self.transforms:
            image,image_all, target = t(image,image_all,target)
        return image,image_all,target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
