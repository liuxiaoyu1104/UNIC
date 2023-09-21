# '/mnt/disk10T/liuxiaoyu/image_crop/GAIC2/annotations/test_4.json'
import json
import glob
import os
import torch
import numpy as np


def compute_iou(gt_crop, pre_crop):
    # print(gt_crop.shape)
    # print(pre_crop.shape)
    zero_t  = np.zeros(gt_crop.shape[0])
    over_x1 = np.maximum(gt_crop[:,0], pre_crop[:,0])
    over_y1 = np.maximum(gt_crop[:,1], pre_crop[:,1])
    over_x2 = np.minimum(gt_crop[:,2], pre_crop[:,2])
    over_y2 = np.minimum(gt_crop[:,3], pre_crop[:,3])
    over_w  = np.maximum(zero_t, over_x2 - over_x1)
    over_h  = np.maximum(zero_t, over_y2 - over_y1)
    inter   = over_w * over_h
    area1   = (gt_crop[:,2] - gt_crop[:,0]) * (gt_crop[:,3] - gt_crop[:,1])
    area2   = (pre_crop[:,2] - pre_crop[:,0]) * (pre_crop[:,3] - pre_crop[:,1])
    union   = area1 + area2 - inter
    iou     = inter / union
    print(iou)
    iou = np.max(iou)

    return iou

def prepare(image_id, anno,h,w):
    image_id = int(image_id)
    image_id = torch.tensor([image_id])

    # print(anno)
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

   
    return  target


iou_90_5_list =[]
iou_85_5_list=[]

with open('/mnt/disk10T/liuxiaoyu/image_crop/GAIC2/annotations/test_4.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)

txt_base = '/mnt/disk10T/liuxiaoyu/image_crop_1/ablation_study_step/ConditionalDETR-image-crop-outside-outpainting-cae+eccv-ema-epoch20-2/txt/7-10-3-4-0.7-gt_max/step_crop'
path_list = glob.glob(os.path.join(txt_base,'*txt'))
for i in range(len(path_list)):
    path_name = path_list[i]
    init_crop =[]
    with open(path_name ,'r') as f:
        for line in f:
            init_crop.append(eval(line))
    if len(init_crop)==0:
        continue 

    image_idx = path_name.split('/')[-1].split('_')[-2]


    # for  k in range(len(json_data['annotations'] )):
    #     print(json_data['annotations'][i]['id'])
    annotations = [obj for obj in json_data['annotations'] if str(obj['image_id']) == image_idx]
    image = [obj for obj in json_data['images'] if str(obj['id']) == image_idx][0]
    height = image['height']
    width = image['width']

    target = prepare(image_idx, annotations,height,width)


    #获得GT
    
    topk_values, score_index = torch.topk(target['scores'].unsqueeze(-1), 5, dim=0)
    gt_boxes = torch.gather( target['boxes'], 0, score_index.repeat( 1, 4))

    crop = np.array(init_crop)  
    # if len(init_crop)<2:
    #     print(image_idx)
    # print(crop.shape)
    iou_max =0 
    # for k in range(crop.shape[0]):
    print(image_idx)
    for k in range(1):
    # for k in range(crop.shape[0]-1):
    
        iou = compute_iou(gt_boxes.numpy(),crop[k+1:k+2])
        if iou>iou_max:
            iou_max =iou
        if k == 0:
            break
    iou_90_5_list.append(int(iou_max>=0.90))
    iou_85_5_list.append(int(iou_max>=0.85))
    # target = prepare(json_data)
    # target = prepare(json_data)
   


print(len(iou_90_5_list))
print(np.mean(iou_90_5_list),np.mean(iou_85_5_list))

        




