import torchvision
import torchvision.transforms as T
import torch
import numpy as np
import cv2
import requests



model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)



class Model:
    def __init__(self,confidence_thresh=0.6):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval();
        self.transform = T.Compose([T.ToTensor()])
        self.conf_thresh = confidence_thresh
        

    def get_seg_output(self,image:np.array):
        image = self.transform(image.copy())
        print(image.shape)
        with torch.no_grad():
            pred = self.model([image])
            
        outputs = [(pred[0]['masks'][i][0],pred[0]['labels'][i]) for i in range(len(pred[0]['boxes'])) if pred[0]['scores'][i]>self.conf_thresh and pred[0]['labels'][i]==1]
#         outputs = [(pred[0]['masks'][i][0],pred[0]['labels'][i]) for i in range(len(pred[0]['boxes'])) if pred[0]['scores'][i]>self.conf_thresh]
        
        return outputs
        
        

class Preprocessing:
    def __init__(self,kernel,lower_bound=0.1,upper_bound=0.9,dilate_iter=10,erode_iter=10):
        self.kernel = kernel
        self.low_thresh = lower_bound
        self.high_thresh = upper_bound
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
        foreground = target_mask >= self.high_thresh
        ambiguous = (target_mask < self.high_thresh)*(target_mask>=self.low_thresh) 
        print(self.erode_iter)
        erode = cv2.erode(foreground.astype('uint8'),self.kernel,iterations=self.erode_iter)
        dilate = cv2.dilate(ambiguous.astype('uint8'),self.kernel,iterations=self.dilate_iter)
        h, w = target_mask.shape
        
        bg_giver = np.clip((erode + dilate),0,1 )
        trimap = np.zeros((h, w, 2))
        trimap[erode == 1, 1] = 1
        trimap[bg_giver == 0, 0] = 1
        
        return trimap
        
        
        


