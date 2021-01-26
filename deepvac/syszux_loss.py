# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn


class LossBase(object):
    def __init__(self, deepvac_config):
        self.auditConfig()

    def auditConfig(self):
        raise Exception("Not implemented!")

    def __call__(self,img):
        raise Exception("Not implemented!")


class BCEBlurWithLogitsLoss(LossBase):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, deepvac_config):
        super(BCEBlurWithLogitsLoss, self).__init__(deepvac_config)
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction=self.reduction)

    def auditConfig(self):
        self.alpha = 0.05
        self.reduction = 'none'

    def __call__(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        # reduce only missing label effects
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(LossBase):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, deepvac_config):
        super(FocalLoss, self).__init__()
        # must be nn.BCEWithLogitsLoss()
        self.loss_fcn = deepvac_config.loss_fcn

    def auditConfig(self):
        self.gamma = 1.5
        self.alpha = 0.25
        self.reduction = 'none'

    def __call__(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QFocalLoss(LossBase):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, deepvac_config):
        super(QFocalLoss, self).__init__()
        # must be nn.BCEWithLogitsLoss()
        self.loss_fcn = deepvac_config.loss_fcn

    def auditConfig(self):
        self.gamma = 1.5
        self.alpha = 0.25
        self.reduction = 'none'

    def __call__(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class Yolov5Loss(LossBase):
    def __init__(self, deepvac_config):
        super(Yolov5Loss, self).__init__(deepvac_config)
        self.cls = deepvac_config.cls
        self.box = deepvac_config.box
        self.obj = deepvac_config.obj
        self.device = deepvac_config.device
        self.strides = torch.Tensor(deepvac_config.strides)
        # define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.cls_pw], device=self.device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.obj_pw], device=self.device))
        if self.fl_gamma > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, self.fl_gamma), FocalLoss(BCEobj, self.fl_gamma)
        self.BCEcls, self.BCEobj = BCEcls, BCEobj
        # model info
        det = deepvac_config.model.detect
        self.ssi = (self.strides == 16).nonzero(as_tuple=False).item()
        for k in ("anchor_num", "class_num", "detect_layer_num", "anchors"):
            setattr(self, k, getattr(det, k))

    def auditConfig(self):
        self.gr = 1.0
        self.anchor_t = 4
        self.cls_pw = 1.0
        self.obj_pw = 1.0
        self.fl_gamma = 0.0
        self.autobalance = False
        self.balance = [4.0, 1.0, 0.4]
        self.cp, self.cn = self.smoothBCE(eps=0.0)

    def build_target(self, pred, target):
        target_num = target.size(0)
        tcls, tbox, indices, anch = [], [], [], []
        # normalized to gridspace gain
        gain = torch.ones(7, device=self.device)
        # targets: (N, 6) -> (3, N, 7), at last append anchor index
        target = [torch.cat([target, i.repeat(target_num).view(target_num, -1)], dim=1) for i in torch.arange(self.anchor_num, device=self.device)]
        target = torch.stack(target, dim=0)

        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device) * 0.5
        for i in range(self.detect_layer_num):
            anchors = self.anchors[i]
            # pred: [(n, c, h1, w1, 9), (n, c, h2, w2, 9), (n, c, h3, w3, 9)] gain: [1, 1, w, h, w, h, 1]
            gain[2:6] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]
            # target: [[[id, cls, cx, cy, w, h, anchor_id], ...n...]]
            # normalize by w, h -> origin size
            t = target * gain
            if target_num:
                # anchors: (3, 2) -> (3, 1, 2)
                wh_radio = t[:, :, 4:6] / anchors.unsqueeze(dim=1)
                # index: (3, N) t: (3, N, 7) -> (n, 7)
                index = torch.max(wh_radio, 1/wh_radio).max(dim=2)[0] < self.anchor_t
                t = t[index]
                # cxcy: [[cx, xy], ...n...]
                cxcy = t[:, 2:4]
                # inverse_cxcy: [[w-cx, h-cy], ...n...]
                inverse_cxcy = gain[[2, 3]] - cxcy
                # cx_index: x_index  cy_index: y_index
                cx_index, cy_index = ((cxcy.fmod(1.) < 0.5) & (cxcy > 1.)).T
                # inverse_cx_index: x_index  inverse_cy_index: y_index
                inverse_cx_index, inverse_cy_index = ((inverse_cxcy % 1. < 0.5) & (inverse_cxcy > 1.)).T
                # cx_index: (n) -> (5, n)
                cx_index = torch.stack((torch.ones_like(cx_index), cx_index, cy_index, inverse_cx_index, inverse_cy_index))
                # t: (n, 7) -> (5, n, 7) -> (n', 7)
                t = t.unsqueeze(dim=0).repeat((5, 1, 1))[cx_index]
                offsets = (torch.zeros_like(cxcy.unsqueeze(dim=0)) + off.unsqueeze(dim=1))[cx_index]
                # offsets = (torch.zeros_like(cxcy)[None] + off[:, None])[cx_index]
            else:
                t = targets[0]
                offsets = 0

            img_id, cls = t[:, :2].long().T
            cxcy = t[:, 2:4]
            wh = t[:, 4:6]
            cxcy_index = (cxcy - offsets).long()
            cx_index, cy_index = cxcy_index.T

            anchor_index = t[:, 6].long()
            indices.append((img_id, anchor_index, cy_index.clamp_(0, gain[3] - 1), cx_index.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((cxcy - cxcy_index, wh), 1))
            anch.append(anchors[anchor_index])
            tcls.append(cls)
        return tcls, tbox, indices, anch

    def __call__(self, pred, target):
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_target(pred, target)

        # pred: [(n, c, h1, w1, 9), (n, c, h2, w2, 9), (n, c, h3, w3, 9)]
        for i, p in enumerate(pred):
            img_id, anchor_index, cy_index, cx_index = indices[i]
            tobj = torch.zeros_like(p[..., 0], device=self.device)
            target_num = img_id.size(0)
            if target_num:
                # p: [px, py, pw, ph, conf, cls] ...
                ps = p[img_id, anchor_index, cy_index, cx_index]
                # Regression
                pcxcy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pcxcy, pwh), 1)
                iou = self.bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()
                # Objectness
                tobj[img_id, anchor_index, cy_index, cx_index] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)
                # Classification
                if self.class_num > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=self.device)
                    t[range(target_num), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE
            lobj += self.BCEobj(p[..., 4], tobj) * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / lobj.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.box
        lobj *= self.obj
        lcls *= self.cls
        bs = tobj.shape[0]

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
        box2 = box2.T
        if x1y1x2y2:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union
        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            if CIoU or DIoU:
                c2 = cw ** 2 + ch ** 2 + eps
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                if DIoU:
                    return iou - rho2 / c2
                elif CIoU:
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / ((1 + eps) - iou + v)
                    return iou - (rho2 / c2 + v * alpha)
            else:
                c_area = cw * ch + eps
                return iou - (c_area - union) / c_area
        else:
            return iou

    @staticmethod
    def smoothBCE(eps=0.1):
        return 1.0 - 0.5 * eps, 0.5 * eps


if __name__ == "__main__":
    import torch

    from config import config

    loss_fn = Yolov5Loss(config)
    net = None
    res = torch.load("example.pkl")
    pred = res["pred"]
    target = res["target"]
    loss = loss_fn(pred, target.to(pred[0].device), net)
    print("loss", loss)

