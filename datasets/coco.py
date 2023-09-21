# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import math
import torch
import torch.utils.data
import torchvision
from util.box_ops import box_cxcywh_to_xyxy, box_iou,box_inter
# from pycocotools import mask as coco_mask
import torch.nn as nn

import datasets.transforms as T
import datasets.sltransform as SLT
import random
from util.box_ops import box_xyxy_to_cxcywh
import torchvision.transforms.functional as F
import numpy as np
from models.QueryQTR.VITGen import TransGen
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from copy import deepcopy
from einops import rearrange
import cv2
import os
import glob


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks,outside_ratio,split):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.__transforms_before = make_coco_tf(split)
        super(CocoDetection, self).__init__(img_folder, ann_file)
        
        self.prepare = ConvertCocoPolysToMask(return_masks,val=False)
        self.outside_ratio = outside_ratio
        self.split = split
        self.img_crop_init = {}
        self.img_crop_image_index = {}
        self.idx_gt=[]
        
        txt_path = './init_view'
        txt_list = glob.glob(os.path.join(txt_path,'init_crop',self.split+'*.txt'))
        for i in range(len(txt_list)):
            idx_name = txt_list[i].split('.')[1].split("_")[-1]
            image_idx_name = txt_list[i].split('.')[1].split("_")[-2]
            init_crop = []
            with open(txt_list[i] ,'r') as f:
                for line in f:
                    init_crop.append(eval(line))   
            self.img_crop_init[idx_name] = init_crop
            self.img_crop_image_index[image_idx_name] =init_crop


        with open(os.path.join(txt_path,"idx_"+str(split)+".txt"),"r") as f:
            for line in f:
                self.idx_gt.append(int(eval(line)))   
        

        

    def __getitem__(self, idx):
        if self.split =='val':
            idx_real = idx 
            image_id = self.ids[idx]
            init_crop = self.img_crop_image_index[str(image_id)]
        else:
            idx_real = idx %(len(self.idx_gt))
            idx_real = self.idx_gt[idx_real]
            init_crop = self.img_crop_init[str(idx_real)]

        img, target = super(CocoDetection, self).__getitem__(idx_real)
        image_id = self.ids[idx_real]
       

        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        

        if self.__transforms_before is not None:
            img, target = self.__transforms_before(img, target)

        W,H=img.size


      
        target['small_sign']= torch.as_tensor([0.0, 0.0,0.0, 0.0])
        target['outpainting']= torch.as_tensor([0, 0,0, 0])
        target["ratio"] = torch.as_tensor([1.0, 1.0])

        #得到 得分最高的 GT
        score_index = torch.argmax(target['scores'])
        gt_max = target['boxes'][score_index:(score_index+1),:]

 
        if self.outside_ratio: 

            x_start ,y_start ,x_end,y_end = random.choice(init_crop)
            x_start = int(x_start)
            y_start = int(y_start)
            x_end = int(x_end)
            y_end = int(y_end)
   

            width = x_end - x_start
            height = y_end - y_start


            for m in range(target["boxes"].shape[0]):
                if target["gt_flags"][m]==1:
                    init_box = torch.from_numpy(np.array([x_start, y_start, x_end, y_end]))
                    iou_item = box_iou(init_box.unsqueeze(0),target["boxes"][m:(m+1),:])[0]

                    if iou_item <0.70:
                        target["gt_flags"][m]=0
        
            
            img_center = F.crop(img,y_start, x_start, height, width) 

            target["boxes"][:,0]= target["boxes"][:,0] - x_start
            target["boxes"][:,1]= target["boxes"][:,1] - y_start
            target["boxes"][:,2]= target["boxes"][:,2] - x_start
            target["boxes"][:,3]= target["boxes"][:,3] - y_start

            target['small_sign']  =  torch.tensor([float(x_start),float(y_start),float(x_end),float(y_end)])
            target["orig_size"] = torch.as_tensor([height, width])
            target["small_sign_in_all"] = target['small_sign']/torch.as_tensor([float(W),float(H),float(W),float(H)]) 
            
        else:
            img_center = img

        if self._transforms is not None:
            img,img_teacher, target = self._transforms(img_center,img, target)
        

        return img,img_teacher, target






class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False,val=False):
        self.return_masks = return_masks
        self.val =val
        

    def __call__(self, image, target):
        w, h = image.size
        

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        scores = [obj["score"] for obj in anno]
        scores = torch.tensor(scores, dtype=torch.float32)

        gt_flags = [obj["gt_flag"] for obj in anno]
        gt_flags = torch.tensor(gt_flags, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        scores = scores[keep]
        gt_flags = gt_flags[keep]

        target = {}
        
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["scores"] = scores
        target["gt_flags"] = gt_flags

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_coco_tf(image_set):
    if image_set == 'train':
         return None
    if image_set == 'val':
        return None
    
def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [384,480,576,672,768,864]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=13330),
            
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([864], max_size=13330),
            normalize,
        ])

    # raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'

    if 'GAIC' in args.coco_path:
        PATHS = {
        "train": (root / "images"/ "train", root / "annotations" / "instances_train.json"),
        "val": (root  / "images" / "test", root / "annotations" / "instances_test.json"),}


    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,outside_ratio=args.outside_ratio,split=image_set)
    return dataset
