import torch
import torch.nn as nn
from focal_loss.focal_loss import FocalLoss
from segmentation_models_pytorch.losses import DiceLoss


def seg_criterion(mask_logits , gt_mask):
    
    bce_loss = nn.BCEWithLogitsLoss()(mask_logits , gt_mask)
    dice_loss = DiceLoss(mode='binary', from_logits=True)(mask_logits , gt_mask)
    
    return bce_loss + dice_loss 


def det_criterion(detections , gt , weight = [1 , 1 , 1]):

    pred_cls , pred_box , pred_center = detections 
    gt_cls , gt_box , gt_center = gt
    
    cls_criterion = torch.nn.CrossEntropyLoss()(pred_cls, gt_cls.long())
    detection_criterion = nn.L1Loss()(pred_box  , gt_box)
    center_criterion = nn.BCEWithLogitsLoss()(pred_center , gt_center)
    
    Wclass , Wdet , Wcenter = weight
    cls_loss = Wclass * cls_criterion
    det_loss = Wdet * detection_criterion
    center_loss = Wcenter * center_criterion 
    
    acc_loss = cls_loss + det_loss + center_loss 
    losses = [cls_loss , det_loss , center_loss]
    
    return acc_loss , losses
    