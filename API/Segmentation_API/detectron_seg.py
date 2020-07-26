from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
import numpy as np
import cv2




class Model:
    def __init__(self,confidence_thresh=0.6):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_thresh  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.model = DefaultPredictor(cfg)


    def get_seg_output(self,image:np.array):
        out = self.model(image)['instances']
        
        outputs = [(out.pred_masks[i],out.pred_classes[i]) for i in  range(len(out.pred_classes)) if out.pred_classes[i]==0]
        
        return outputs
    
    

class Preprocessing:
    def __init__(self,kernel,dilate_iter=5,erode_iter=1):
        self.kernel = kernel
        self.dilate_iter = dilate_iter
        self.erode_iter = erode_iter
        
    def get_target_mask(self,masks):
        out = np.zeros(masks[0].shape)
        for mask in masks:
            out += mask
        out = np.clip(out,0,1)
        return out

    def get_trimap(self,masks):
        target_mask = self.get_target_mask(masks)
        erode = cv2.erode(target_mask.astype('uint8'),self.kernel,iterations=self.erode_iter)
        dilate = cv2.dilate(target_mask.astype('uint8'),self.kernel,iterations=self.dilate_iter)
        h, w = target_mask.shape
         
        trimap = np.zeros((h, w, 2))
        trimap[erode == 1, 1] = 1
        trimap[dilate == 0, 0] = 1
        
        return trimap
        
        
