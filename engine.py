# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
from operator import index
import os
import sys
from typing import Iterable
import math

import torch
from util.box_ops import box_cxcywh_to_xyxy, box_iou
import numpy as np

import util.misc as utils
import glob
from datasets.panoptic_eval import PanopticEvaluator
import cv2
import matplotlib.pyplot as plt
import os
from einops import rearrange


def denorm_img(tensor):
    tensor = rearrange(tensor[0:4], 'b c w h -> b w h c').detach().cpu()
    # tensor = tensor * torch.tensor((0.229, 0.224, 0.225)) + torch.tensor((0.485, 0.456, 0.406))
    tensor = np.clip(tensor.flatten(0, 1).numpy(), 0, 1)
    return tensor

def train_one_epoch(model: torch.nn.Module,ema: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    label_class: str='soft',outpainting: bool=False,):
    model.train()
    ema.eval()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    count =0 
    if epoch>=20:
        label_class ='SD'
    # if epoch>=40:
    #     label_class ='onlySD'
    print(label_class)
    for samples,samples_all,targets in metric_logger.log_every(data_loader, print_freq, header):
        count +=1

        samples = samples.to(device)
        samples_all =samples_all.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        
        outputs = model(samples,True,label_class)  
        with torch.no_grad():
            outputs_all = ema(samples_all,False,label_class)  
    
        loss_dict = criterion(outputs, targets, outputs_all,label_class)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict )

        
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        

        
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        optimizer.zero_grad()
        
      
        ema.to(device)
        ema.update()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model ,ema, criterion, postprocessors, data_loader, device, output_dir,out_num,print_pic,dataset_path,outpainting,outside_ratio):
  
    model.eval()
    ema.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1,fmt="{global_avg_true:.4f}"))
    metric_logger.add_meter('class_error_85', utils.SmoothedValue(window_size=1,fmt="{global_avg_true:.4f}"))
    metric_logger.add_meter('class_error_75', utils.SmoothedValue(window_size=1,fmt="{global_avg_true:.4f}"))
    metric_logger.add_meter('class_error_70', utils.SmoothedValue(window_size=1,fmt="{global_avg_true:.4f}"))
    header = 'Test:'

    count =0 
    ratio_list_1 =[]
    ratio_list_2 =[]
    for samples,samples_all, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        samples_all = samples_all.to(device)
        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # outputs = model(samples,outpainting,outside_ratio)
        outputs = model(samples,True,'soft')
        outputs_all = ema(samples_all,False,'soft')
        loss_dict = criterion(outputs, targets,outputs_all,'soft')
        weight_dict = criterion.weight_dict

        
            
        count += 1


        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(class_error_85=loss_dict_reduced['class_error_85'])
        metric_logger.update(class_error_75=loss_dict_reduced['class_error_75'])
        metric_logger.update(class_error_70=loss_dict_reduced['class_error_70'])
       

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        orig_target_starts = torch.stack([t["small_sign"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes,orig_target_starts)
        
        #保存图片
        if print_pic:
            #图片的保存路径
            output_path = os.path.join(output_dir,'result')
            os.makedirs(output_path, exist_ok=True)
            path = os.path.join(dataset_path,'images','test')
            
                
            index = 0 
            
            for target, output in zip(targets,results):


                ratio_list_1.append(target['ratio'][0].cpu().numpy())
                ratio_list_2.append(target['ratio'][1].cpu().numpy())

                #get output
                pre_boxes = output['boxes']
                # get best gt

                scale_fct = torch.stack([target['orig_size'][1], target['orig_size'][0], target['orig_size'][1], target['orig_size'][0]], dim=0)
                start_fct = torch.stack([target['small_sign'][0], target['small_sign'][1], target['small_sign'][0], target['small_sign'][1]], dim=0)
                init_box = torch.stack([target['small_sign'][0], target['small_sign'][1], target['small_sign'][2], target['small_sign'][3]], dim=0)
                init_W = target['small_sign'][2] -target['small_sign'][0]
                init_H = target['small_sign'][3] -target['small_sign'][1]
            
                init_box = init_box * scale_fct


                if 'FLMS' not in dataset_path:
                    
                    gt_boxes = target['boxes'][target['gt_flags']==1]
                    gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
                    gt_boxes = gt_boxes +start_fct
                    gt_boxes = gt_boxes*scale_fct

                    iou = box_iou(pre_boxes,gt_boxes)[0]
              
                else:
                    iou = 0
                    for ind in range(target['boxes'].shape[0]):
                        gt_boxes_item = target['boxes'][ind:ind+1,:]
                        gt_boxes_item = box_cxcywh_to_xyxy(gt_boxes_item)
                        scale_fct = torch.stack([target['orig_size'][1], target['orig_size'][0], target['orig_size'][1], target['orig_size'][0]], dim=0)
                        gt_boxes_item = gt_boxes_item*scale_fct
                        iou_item = box_iou(pre_boxes,gt_boxes_item)[0]
                        if iou_item > iou:
                            iou = iou_item
                            gt_boxes = gt_boxes_item

                
                # 打印 裁剪后的图 
                image_name=str(target['image_id'][0].cpu().numpy())
                image_path= glob.glob(os.path.join(path,image_name+'.jpg'))
                # print(os.path.join(path,image_name+'.jpg'))
                if len(image_path)==0:
                    image_path = glob.glob(os.path.join(path,'0'+image_name+'_Large.jpg'))
                img = cv2.imread(image_path[0])
              
                
                os.makedirs(output_path, exist_ok=True)
                save_dir = '%s/%s.jpg' % (output_path, image_name+'_crop_out')
                pre_boxes_numpy = pre_boxes[0].cpu().numpy()
                pre_boxes_numpy[1] = np.clip(pre_boxes_numpy[1],0,img.shape[0]-1)
                pre_boxes_numpy[3] = np.clip(pre_boxes_numpy[3],0,img.shape[0]-1)
                pre_boxes_numpy[0] = np.clip(pre_boxes_numpy[0],0,img.shape[1]-1)
                pre_boxes_numpy[2] = np.clip(pre_boxes_numpy[2],0,img.shape[1]-1)
                img_crop =  img[int(pre_boxes_numpy[1]):int(pre_boxes_numpy[3]),int(pre_boxes_numpy[0]):int(pre_boxes_numpy[2]),:]
                cv2.imwrite(save_dir, img_crop)
                
                #print input
                save_dir = '%s/%s.jpg' % (output_path, image_name+'_input')
                image_input = samples.tensors[index].cpu().permute(1, 2, 0).numpy()* 255
                cv2.imwrite(save_dir, image_input)

               
                cv2.rectangle(img, (int(init_box[0]),int(init_box[1])),
                (int(init_box[2]-4),int(init_box[3]-4)), (0, 0, 255), 1)

                save_dir = '%s/%s.jpg' % (output_path, str(image_name)+'_gt_pre_best_out')

                for j in range(gt_boxes.shape[0]):
                    gt_boxes_numpy  = gt_boxes[j].cpu().numpy()
                    cv2.rectangle(img, (int(gt_boxes_numpy[0]),int(gt_boxes_numpy[1])),
                    (int(gt_boxes_numpy[2]),int(gt_boxes_numpy[3])), (255, 0, 0), 3)

                
                img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_REPLICATE)
                for m in range(1):
                    pre_boxes_numpy= pre_boxes[m].cpu().numpy()
                    pre_boxes_numpy = pre_boxes_numpy +50
                    pre_boxes_numpy[1] = np.clip(pre_boxes_numpy[1],0,img.shape[0]-1)
                    pre_boxes_numpy[3] = np.clip(pre_boxes_numpy[3],0,img.shape[0]-1)
                    pre_boxes_numpy[0] = np.clip(pre_boxes_numpy[0],0,img.shape[1]-1)
                    pre_boxes_numpy[2] = np.clip(pre_boxes_numpy[2],0,img.shape[1]-1)

                    cv2.rectangle(img, (int(pre_boxes_numpy[0]),int(pre_boxes_numpy[1])), 
                    (int(pre_boxes_numpy[2]),int(pre_boxes_numpy[3])), (0, 255, 0), 2)
                cv2.imwrite(save_dir, img)


                index +=1
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if 'class_error' not in k}
    stats_test_class_error = {k: meter.global_avg_true for k, meter in metric_logger.meters.items() if 'class_error'  in k}
    stats.update(stats_test_class_error)

    return stats
