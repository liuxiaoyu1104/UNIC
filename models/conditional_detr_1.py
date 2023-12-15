# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------



import math
import torch
import torch.nn.functional as F
from xmlrpc.client import Boolean
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy_best, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
import cv2


class ConditionalDETR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        



    def forward(self, samples: NestedTensor, part:Boolean):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()

        
        assert mask is not None
        hs, reference,memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1],part)
        
        if part==False:
            for x in range(len(memory)):
                memory[x] = memory[x].detach()
            out = {'memory':memory}
            return out
     
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed(hs[lvl])
            tmp[..., :2] = tmp[..., :2]+reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)

        
        outputs_class = self.class_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],'memory':memory}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
       
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

def get_score(map, score,gt_flag):
    # map [90,86]  score[1,86,1]
    score= score.unsqueeze(0).unsqueeze(-1)
    saliency_score = torch.cat([score[:,torch.argmax(map[l]):torch.argmax(map[l])+1] for l in range(map.shape[0])], dim=1)
    saliency_score[saliency_score < 2] = 0 
    saliency_score[(saliency_score >= 2) & (saliency_score < 3.5)] = 0.5 * (
        saliency_score[(saliency_score >= 2) & (saliency_score < 3.5)] - 2) / (3.5 - 2)
    saliency_score[saliency_score >= 3.5] = 0.5
    
    map_gt = torch.permute(map, ( 1, 0))[gt_flag==1]
    map_gt = torch.max(map_gt,dim=0)[0]
    saliency_score = saliency_score.squeeze(0).squeeze(-1)
    saliency_score[map_gt>0.9] = saliency_score[map_gt>0.9] 
    return saliency_score.unsqueeze(0).unsqueeze(-1)
    # return saliency_score

def label_smooth(pre_box, targets):
    # pre_box [300,1]
    bs, num_queries = pre_box.shape[:2]
    tgt_bbox = torch.cat([t['boxes'] for t in targets], dim=0)  # 3*87*4 -> 261*4
    out_bbox = pre_box.flatten(0, 1) # 3*90*4=270*4
    cost_giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
    C = cost_giou.view(bs, num_queries, -1) # 270* 261    3*90*261

    sizes = [len(v["boxes"]) for v in targets] # 80n87
    smoothed_label = torch.cat([get_score(c[i], targets[i]['scores'], targets[i]['gt_flags']) for i, c in enumerate(C.split(sizes, -1))], dim=0)

    return smoothed_label

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        

    def loss_labels(self, outputs, targets_all,outputs_ema, indices, num_boxes, log=True,label_class='soft'):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

       
        if label_class=='soft':
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][t["gt_flags"]==1][J] for t, (_, J) in zip(targets_all, indices)])
            target_classes = label_smooth(outputs['pred_boxes'],targets_all) 
            target_classes[idx] = 1
            loss_ce = sigmoid_focal_loss(src_logits, target_classes, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                    src_logits.shape[1]
            # loss_ce = sigmoid_focal_loss(src_logits, target_classes, num_boxes, alpha=self.focal_alpha, gamma=2) 
            losses = {'loss_ce': loss_ce}
           

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # losses['class_error'] = 100
            losses['class_error'] =  accuracy_best(outputs, targets_all,iou_threshold=0.90)[0]
            losses['class_error_85'] =  accuracy_best(outputs, targets_all,iou_threshold=0.85)[0]
            losses['class_error_75'] =  accuracy_best(outputs, targets_all,iou_threshold=0.75)[0]
            losses['class_error_70'] =  accuracy_best(outputs, targets_all,iou_threshold=0.70)[0]
        return losses

    
    def loss_tokens(self, outputs, targets_all,outputs_ema, indices, num_boxes):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'memory' in outputs
        loss_token_ema =0 
        token_pre = outputs['memory']
        token_gt = outputs_ema['memory']
        for i in range(len(token_pre)):
            token_pre_item = token_pre[i]
            token_gt_item = token_gt[i]
            b,c,H,W = token_gt_item.shape # torch.Size([1, 256, 33, 49])
            b,c,h,w = token_pre_item.shape #torch.Size([1, 256, 42, 62])
            #获得小图
            x_start,y_start,x_end,y_end= targets_all[i]['outpainting'][:]
            
            x_start = int(x_start);y_start = int(y_start);
            x_end=int(x_end);y_end=int(y_end)
            # 3 1 47 31  width 44  height 30
            width = x_end - x_start
            height = y_end - y_start
            #获得 图外的补全的边缘
            dis_new_left =round(0.2*width) # 9 
            dis_new_right =round(0.2*width) 
            dis_new_top =round(0.2*height) # 6 
            dis_new_bottom =round(0.2*height)
            #获得应该外边缘
            left_min = min(dis_new_left,x_start)
            right_min = min(dis_new_right,W-x_end)
            top_min = min(dis_new_top,y_start)
            bottom_min = min(dis_new_bottom,H-y_end)


            start_x_ori = x_start - left_min
            end_x_ori = x_end + right_min
            start_y_ori = y_start - top_min
            end_y_ori = y_end + bottom_min
            
            token_gt_item_part = token_gt_item[:,:,start_y_ori:end_y_ori,start_x_ori:end_x_ori]

            start_x_pre = dis_new_left - left_min
            end_x_pre = w - dis_new_right + right_min
            start_y_pre = dis_new_top - top_min
            end_y_pre = h -dis_new_bottom + bottom_min

            token_pre_item_part = token_pre_item[:,:,start_y_pre:end_y_pre,start_x_pre:end_x_pre]

            
        

            # for  k  in range(token_gt_item_part.shape[1]):
                
            #     token_gt_item_part_c = token_gt_item_part[0,k]
            #     token_gt_item_part_c = torch.abs(token_gt_item_part_c)
            #     token_gt_item_part_c = (token_gt_item_part_c-torch.min(token_gt_item_part_c))/(torch.max(token_gt_item_part_c)-torch.min(token_gt_item_part_c))
            #     save_dir = '%s/%s.jpg' % ('./test/mask', str(targets_all[i]['image_id'].cpu().numpy()[0])+'_'+str(k)+'_gt_part')
            #     image_input = token_gt_item_part_c.cpu().numpy()* 255
               
            #     cv2.imwrite(save_dir, image_input)


            #     token_gt_item_small_c = token_gt_item[0,k,y_start:y_end,x_start:x_end]
            #     token_gt_item_small_c = torch.abs(token_gt_item_small_c)
            #     token_gt_item_small_c = (token_gt_item_small_c-torch.min(token_gt_item_small_c))/(torch.max(token_gt_item_small_c)-torch.min(token_gt_item_small_c))
            #     save_dir = '%s/%s.jpg' % ('./test/mask', str(targets_all[i]['image_id'].cpu().numpy()[0])+'_'+str(k)+'_gt_small')
            #     image_input = token_gt_item_small_c.cpu().numpy()* 255
            #     cv2.imwrite(save_dir, image_input)


            #     token_gt_item_c = token_gt_item[0,k]
            #     token_gt_item_c = torch.abs(token_gt_item_c)
            #     token_gt_item_c = (token_gt_item_c-torch.min(token_gt_item_c))/(torch.max(token_gt_item_c)-torch.min(token_gt_item_c))
            #     save_dir = '%s/%s.jpg' % ('./test/mask', str(targets_all[i]['image_id'].cpu().numpy()[0])+'_'+str(k)+'_gt')
            #     image_input = token_gt_item_c.cpu().numpy()* 255
            #     cv2.rectangle(image_input, (int(x_start),int(y_start)), 
            #         (int(x_end),int(y_end)), (0, 255, 0), 1)
            #     cv2.rectangle(image_input, (int(start_x_ori),int(start_y_ori)), 
            #         (int(end_x_ori),int(end_y_ori)), (0, 0, 210), 1)

            #     cv2.imwrite(save_dir, image_input)

            #     token_pre_item_part_c = token_pre_item_part[0,k]
            #     token_pre_item_part_c = torch.abs(token_pre_item_part_c)
            #     token_pre_item_part_c = (token_pre_item_part_c-torch.min(token_pre_item_part_c))/(torch.max(token_pre_item_part_c)-torch.min(token_pre_item_part_c))
            #     save_dir = '%s/%s.jpg' % ('./test/mask', str(targets_all[i]['image_id'].cpu().numpy()[0])+'_'+str(k)+'_pre_part')
            #     image_input = token_pre_item_part_c.detach().cpu().numpy()* 255
            #     cv2.imwrite(save_dir, image_input)


            #     token_pre_item_small_c = token_pre_item[0,k,dis_new_top:-dis_new_top,dis_new_left:-dis_new_left]
            #     token_pre_item_small_c = torch.abs(token_pre_item_small_c)
            #     token_pre_item_small_c = (token_pre_item_small_c-torch.min(token_pre_item_small_c))/(torch.max(token_pre_item_small_c)-torch.min(token_pre_item_small_c))
            #     save_dir = '%s/%s.jpg' % ('./test/mask', str(targets_all[i]['image_id'].cpu().numpy()[0])+'_'+str(k)+'_predict_small')
            #     image_input = token_pre_item_small_c.detach().cpu().numpy()* 255
                
            #     cv2.imwrite(save_dir, image_input)


            #     token_pre_item_c = token_pre_item[0,k]
            #     token_pre_item_c = torch.abs(token_pre_item_c)
            #     token_pre_item_c = (token_pre_item_c-torch.min(token_pre_item_c))/(torch.max(token_pre_item_c)-torch.min(token_pre_item_c))
            #     save_dir = '%s/%s.jpg' % ('./test/mask', str(targets_all[i]['image_id'].cpu().numpy()[0])+'_'+str(k)+'_predict')
            #     image_input = token_pre_item_c.detach().cpu().numpy()* 255
            #     cv2.rectangle(image_input, (int(dis_new_left),int(dis_new_top)), 
            #         (int(w - dis_new_right),int(h -dis_new_bottom)), (0, 255, 0), 1)
            #     cv2.rectangle(image_input, (int(start_x_pre),int(start_y_pre)), 
            #         (int(end_x_pre),int(end_y_pre)), (0, 0, 210), 1)

            #     cv2.imwrite(save_dir, image_input)


            # mask = torch.ones_like(token_gt_item_part, dtype=torch.bool, device=token_pre_item.device)
            # b,c,h_real,w_real = token_gt_item_part.shape
            # mask[:,:,top_min:h_real-bottom_min,left_min:w_real-right_min] = False

            # token_gt_item_part = token_gt_item_part.flatten(2)[0]
            # token_pre_item_part = token_pre_item_part.flatten(2)[0]
            # mask = mask.flatten(2)[0,0]
            # token_gt_item_part = torch.transpose(token_gt_item_part,0,1)
            # token_pre_item_part = torch.transpose(token_pre_item_part,0,1)
            # token_gt_item_part = token_gt_item_part[mask]
            # token_pre_item_part = token_pre_item_part[mask]
            # token_gt_item_part = torch.transpose(token_gt_item_part,0,1)
            # token_pre_item_part = torch.transpose(token_pre_item_part,0,1)

            # mean = token_gt_item_part.mean(dim=-1, keepdim=True)
            # var = token_gt_item_part.var(dim=-1, keepdim=True)
            # token_gt_item_part = (token_gt_item_part - mean) / (var + 1.e-6)**.5
            # mean = token_pre_item_part.mean(dim=-1, keepdim=True)
            # var = token_pre_item_part.var(dim=-1, keepdim=True)
            # token_pre_item_part = (token_pre_item_part - mean) / (var + 1.e-6)**.5


            
            
            loss_token_ema += F.smooth_l1_loss(token_gt_item_part,token_pre_item_part)
            

        losses = {'loss_tokens': loss_token_ema}
       
        return losses

    
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets_all,outputs_ema, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
       

        tgt_lengths = torch.as_tensor([len(v["labels"][v['gt_flags']==1]) for v in targets_all], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets_all,outputs_ema, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]

        
        target_boxes = torch.cat([t['boxes'][t['gt_flags']==1][i] for t, (_, i) in zip(targets_all, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets,outputs_ema, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets,outputs_ema, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'tokens': self.loss_tokens
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets,outputs_ema, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets_all,outputs_ema,label_class):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets_all)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"][t['gt_flags']==1]) for t in targets_all)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == 'labels' and self.training:
                kwargs = {'log': False,'label_class':label_class}
            losses.update(self.get_loss(loss, outputs, targets_all,outputs_ema, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets_all)
                for loss in self.losses:
                    if loss == 'tokens':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False,'label_class':label_class}
                    
                    l_dict = self.get_loss(loss, aux_outputs, targets_all,None, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes,target_start):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 5, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        img_h, img_w = target_sizes.unbind(1)

        start_w, start_h,_,_ = target_start.unbind(1)

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        start_fct = torch.stack([start_w, start_h, start_w, start_h], dim=1)
        # boxes = torch.clamp(boxes,0.0,0.999)

        boxes = boxes + start_fct[:, None, :]
        boxes = boxes * scale_fct[:, None, :]


        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)

        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 1
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = ConditionalDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    # weight_dict = {'loss_ce': args.cls_loss_coef}
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.outpainting:
        weight_dict["loss_tokens"] = args.token_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    # losses = ['labels']
    if args.outpainting:
        losses = losses+["tokens"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    

    return model, criterion, postprocessors
