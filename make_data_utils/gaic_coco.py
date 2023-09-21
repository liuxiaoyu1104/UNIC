#!/usr/bin/python

# pip install lxml

import sys
import os
import json
import xml.etree.ElementTree as ET
import glob
import cv2 

START_BOUNDING_BOX_ID = 1


PRE_DEFINE_CATEGORIES = {"crop": 0}

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))




def convert(xml_files, json_file,split,only_gt):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID

    for txt_file in txt_files:
        filename = os.path.basename(txt_file).split('.')[0]+'.jpg'
    
        image_id = get_filename_as_int(txt_file)
        image = cv2.imread(os.path.join('/mnt/disk10T/liuxiaoyu/image_crop/GAIC2/images',split,filename))
        height,width,C =image.shape
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        
        with open(txt_file, 'r') as fid:
            annotations_txt = fid.readlines()
            annotations_sort = sorted(annotations_txt, key=lambda f: float(f.split()[4]))
            annotation_best = annotations_sort[-1].split()[4]
            #获取整个文件框的最大值
            # print(annotation_best
            if float(annotation_best)>4:
                #先获取到iamge
                json_dict["images"].append(image)
                for annotation in annotations_txt:
                    annotation_split = annotation.split()
                    if float(annotation_split[4]) != -2:
        
                        xmin = int(annotation_split[1])
                        ymin = int(annotation_split[0])
                        xmax = int(annotation_split[3])
                        ymax = int(annotation_split[2])
                        assert xmax > xmin
                        assert ymax > ymin
                        o_width = abs(xmax - xmin)
                        o_height = abs(ymax - ymin)
                        score = float(annotation_split[4])
                        gt_flag = 0
                        if float(annotation_split[4]) > 4 :
                            gt_flag = 1
                        if only_gt and gt_flag==0:
                            continue
                        ann = {
                            "area": o_width * o_height,
                            "image_id": image_id,
                            "bbox": [xmin, ymin, o_width, o_height],
                            "category_id": categories['crop'],
                            "id": bnd_id,
                            "score" :score,
                            "gt_flag" :gt_flag,
                            "iscrowd": 0,
                        }
                        json_dict["annotations"].append(ann)
                        bnd_id = bnd_id + 1

        
    

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Gaic annotation to COCO format."
    )
    path_base = '/mnt/disk10T/liuxiaoyu/image_crop/GAIC2'
    split_list=['train','test','val']
    only_gt =True
    for split in split_list:

        txt_dir = os.path.join(path_base,'annotations',split)
        txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
        if only_gt:
            gt_str='_only_gt'
        else:
            gt_str=''
        json_file = os.path.join(path_base,'annotations','instances_'+ split +gt_str+'_4.json')

        convert(txt_files, json_file,split,only_gt)
        print("Success: {}".format(json_file))