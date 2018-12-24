import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFn:
    def __init__(self, device):
        # loss function
        self.loss_cls = nn.BCELoss().to(device)
        self.loss_box = nn.MSELoss().to(device)
        self.loss_landmark = nn.MSELoss().to(device)


    def cls_loss(self,gt_label,pred_label):
        # get the mask element which >= 0, only 0 and 1 can effect the detection loss
        pred_label = torch.squeeze(pred_label)
        mask = torch.ge(gt_label,0)
        valid_gt_label = torch.masked_select(gt_label,mask).float()
        valid_pred_label = torch.masked_select(pred_label,mask)
        return self.loss_cls(valid_pred_label,valid_gt_label)


    def box_loss(self,gt_label,gt_offset,pred_offset):
        #get the mask element which != 0
        mask = torch.ne(gt_label,0)
        #convert mask to dim index
        chose_index = torch.nonzero(mask)
        chose_index = torch.squeeze(chose_index)
        #only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index,:]
        valid_pred_offset = pred_offset[chose_index,:]
        valid_pred_offset = torch.squeeze(valid_pred_offset)
        return self.loss_box(valid_pred_offset,valid_gt_offset)


    def landmark_loss(self,gt_label,gt_landmark,pred_landmark):
        mask = torch.eq(gt_label,-2)

        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        return self.loss_landmark(valid_pred_landmark, valid_gt_landmark)