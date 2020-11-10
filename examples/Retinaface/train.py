import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F

import cv2
import math
import time
import numpy as np
import os

from deepvac.syszux_log import LOG
from deepvac.syszux_deepvac import DeepvacTrain

from modules.model import RetinaFace, MultiBoxLoss
from modules.utils_prior_box import PriorBox
from modules.utils_nms import py_cpu_nms
from modules.utils_box import decode, decode_landm
from modules.utils_evaluation import image_eval, img_pr_info, dataset_pr_info, voc_ap

from aug.aug import preproc
from synthesis.synthesis import RetinaTrainDataset, RetinaValDataset, detection_collate

class DeepvacRetina(DeepvacTrain):
    def __init__(self, retina_config):
        super(DeepvacRetina, self).__init__(retina_config)
        priorbox = PriorBox(self.conf.cfg, image_size=self.conf.cfg['image_size'])
        with torch.no_grad():
            self.priors = priorbox.forward()
            self.priors = self.priors.to(self.device)
        self.step_index = 0
    
    def initNetWithCode(self):
        self.net = RetinaFace(self.conf.cfg)
        self.net.to(self.conf.device)
        if self.conf.cfg['ngpu'] > 1 and self.conf.cfg['gpu_train']:
            self.net = torch.nn.DataParallel(self.net)
        
    def initScheduler(self):
        pass
 
    def initCriterion(self):
        self.criterion = MultiBoxLoss(self.conf.cls_num, 0.35, True, 0, True, 7, 0.35, False)
    
    def initTrainLoader(self):
        self.train_dataset = RetinaTrainDataset(self.conf.train.label_path, preproc(840, (104, 117, 123)))
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.conf.train.batch_size, num_workers=self.conf.num_workers, shuffle=self.conf.train.shuffle, collate_fn=detection_collate)

    def initValLoader(self):
        self.val_dataset = RetinaValDataset(self.conf)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.conf.val.batch_size, num_workers=self.conf.num_workers, shuffle=self.conf.val.shuffle, collate_fn=detection_collate)

    def initOptimizer(self):
        self.optimizer = optim.SGD(
                self.net.parameters(),
                lr=self.conf.lr,
                momentum=self.conf.momentum,
                weight_decay=self.conf.weight_decay,
        )

    def doLoss(self):
        if not self.is_train:
            return
        self.loss_l, self.loss_c, self.loss_landm = self.criterion(self.output, self.priors, self.target)
        self.loss = self.conf.cfg['loc_weight'] * self.loss_l + self.loss_c + self.loss_landm


    def doForward(self):
        self.output = self.net(self.sample)

    def preEpoch(self):
        if self.is_train:
            if self.epoch in [self.conf.cfg['decay1'], self.conf.cfg['decay2']]:
                self.step_index += 1
            self.adjust_learning_rate()
        else:
            self.face_count = 0
            self.pr_curve = 0

    def earlyIter(self):        
        start = time.time()
        self.sample = self.sample.to(self.device)
        self.target = [anno.to(self.device) for anno in self.target]
        if not self.is_train:
            return    
        self.data_cpu2gpu_time.update(time.time() - start)
        try:
            self.addGraph(self.sample)
        except:
            LOG.logW("Tensorboard addGraph failed. You network foward may have more than one parameters?")
            LOG.logW("Seems you need reimplement preIter function.")

    def preIter(self):
        pass

    def postIter(self):
        if self.is_train:
            return
        loc, conf, landms = self.output
        conf = F.softmax(conf, dim=-1)
        priorbox = PriorBox(self.conf.cfg, image_size=(self.sample.shape[2], self.sample.shape[3]))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        resize = 1.0
        scale = torch.Tensor([self.sample.shape[3], self.sample.shape[2], self.sample.shape[3], self.sample.shape[2]])
        scale = scale.to(self.device)
        boxes = decode(loc.data.squeeze(0), prior_data, self.conf.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.conf.cfg['variance'])
        scale1 = torch.Tensor([self.sample.shape[1], self.sample.shape[3], self.sample.shape[1], self.sample.shape[3],
                        self.sample.shape[1], self.sample.shape[3], self.sample.shape[1], self.sample.shape[3],
                        self.sample.shape[1], self.sample.shape[3]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()
        
        # ignore low scores
        inds = np.where(scores > self.conf.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        
        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.conf.nms_threshold)

        dets = dets[keep, :]
        landms = landms[keep]
        dets_np = dets
        target_np = self.target[0].cpu().numpy()
        ignore = np.ones(target_np.shape[0])
        dets_np = dets_np.astype('float64')
        target_np = target_np.astype('float64')
        pred_recall, proposal_list = image_eval(dets_np, target_np, ignore, 0.5)
        _img_pr_info = img_pr_info(1000, dets_np, proposal_list, pred_recall)
        self.pr_curve += _img_pr_info
        self.face_count += target_np.shape[0]

    def postEpoch(self):
        if self.is_train:
            return
        self.pr_curve = dataset_pr_info(1000, self.pr_curve, self.face_count)
        propose = self.pr_curve[:, 0]
        recall = self.pr_curve[:, 1]
        self.accuracy = voc_ap(recall, propose)
        LOG.logI('Test accuray: {:.4f}'.format(self.accuracy))

    def processAccept(self):
        pass

    def adjust_learning_rate(self):
        warmup_epoch = -1
        if self.epoch <= warmup_epoch:
            self.conf.lr = 1e-6 + (self.conf.lr-1e-6) * self.iter / (math.ceil(len(self.train_dataset)/self.conf.train.batch_size) * warmup_epoch)
        else:
            self.conf.lr = self.conf.lr * (self.conf.gamma ** (self.step_index))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.conf.lr

if __name__ == "__main__":
    from config import config

    dr = DeepvacRetina(config)
    dr(None)
