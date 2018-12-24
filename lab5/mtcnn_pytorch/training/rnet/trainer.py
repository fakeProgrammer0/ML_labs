import torch
import datetime
import time
from models.lossfn import LossFn
from tools.utils import AverageMeter


class RNetTrainer(object):
    
    def __init__(self, lr, train_loader, model, optimizer, scheduler, logger, device):
        self.lr = lr
        self.train_loader = train_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.lossfn = LossFn(self.device)
        self.logger = logger
        self.run_count = 0
        self.scalar_info = {}

    def compute_accuracy(self, prob_cls, gt_cls):
        #we only need the detection which >= 0
        prob_cls = torch.squeeze(prob_cls)
        mask = torch.ge(gt_cls, 0)
        #get valid element
        valid_gt_cls = torch.masked_select(gt_cls, mask)
        valid_prob_cls = torch.masked_select(prob_cls, mask)
        size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
        prob_ones = torch.ge(valid_prob_cls, 0.6).float()
        right_ones = torch.eq(prob_ones, valid_gt_cls.float()).float()

        return torch.div(torch.mul(torch.sum(right_ones), float(1.0)), float(size))

    def update_lr(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        """
        # update learning rate of model optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def train(self, epoch):
        cls_loss_ = AverageMeter()
        box_offset_loss_ = AverageMeter()
        total_loss_ = AverageMeter()
        accuracy_ = AverageMeter()

        self.scheduler.step()
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            gt_label = target['label']
            gt_bbox = target['bbox_target']
            data, gt_label, gt_bbox = data.to(self.device), gt_label.to(
                self.device), gt_bbox.to(self.device).float()

            cls_pred, box_offset_pred = self.model(data)
            # compute the loss
            cls_loss = self.lossfn.cls_loss(gt_label, cls_pred)
            box_offset_loss = self.lossfn.box_loss(
                gt_label, gt_bbox, box_offset_pred)

            total_loss = cls_loss + box_offset_loss * 0.5
            accuracy = self.compute_accuracy(cls_pred, gt_label)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            cls_loss_.update(cls_loss, data.size(0))
            box_offset_loss_.update(box_offset_loss, data.size(0))
            total_loss_.update(total_loss, data.size(0))
            accuracy_.update(accuracy, data.size(0))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                epoch, batch_idx * len(data), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), total_loss.item(), accuracy.item()))

        self.scalar_info['cls_loss'] = cls_loss_.avg
        self.scalar_info['box_offset_loss'] = box_offset_loss_.avg
        self.scalar_info['total_loss'] = total_loss_.avg
        self.scalar_info['accuracy'] = accuracy_.avg
        self.scalar_info['lr'] = self.scheduler.get_lr()[0]

        if self.logger is not None:
            for tag, value in list(self.scalar_info.items()):
                self.logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}
        self.run_count += 1

        print("|===>Loss: {:.4f}".format(total_loss_.avg))
        return cls_loss_.avg, box_offset_loss_.avg, total_loss_.avg, accuracy_.avg