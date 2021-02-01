# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ArcFace(nn.Module):
    def __init__(self, embedding_size, class_num, s=32.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_feature = embedding_size
        self.out_feature = class_num
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.Tensor(self.out_feature, self.in_feature))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.kernel))
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm) if not self.easy_margin else torch.where(cosine > 0, phi, cosine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output


class CurricularFace(nn.Module):
    def __init__(self, embedding_size, class_num, s=64.0, m=0.50):
        super(CurricularFace, self).__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, x, label):
        kernel_norm = F.normalize(self.kernel)
        cos_theta = torch.mm(x, kernel_norm).clamp(-1, 1)
        target_logit = cos_theta[torch.arange(0, x.size(0)), label].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)
        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output


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
        target = [torch.cat([target, i.repeat(target_num).view(target_num, 1)], dim=1) for i in torch.arange(self.anchor_num, device=self.device)]
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
                t = target[0]
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

    def compute_loss(self, p, tcls, tbox, indices, anchors, balance):
        img_id, anchor_index, cy_index, cx_index = indices
        tobj = torch.zeros_like(p[..., 0], device=self.device)
        target_num = img_id.size(0)
        if not target_num:
            lobj = self.BCEobj(p[..., 4], tobj) * balance
            balance = balance * 0.9999 + 0.0001 / lobj.detach().item() if self.autobalance else balance
            return 0, 0, lobj
        # p: [px, py, pw, ph, conf, cls] ...
        ps = p[img_id, anchor_index, cy_index, cx_index]
        # Regression
        pcxcy = ps[:, :2].sigmoid() * 2. - 0.5
        pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors
        pbox = torch.cat((pcxcy, pwh), 1)
        iou = self.bbox_iou(pbox.T, tbox, x1y1x2y2=False, CIou=True)
        lbox = (1.0 - iou).mean()
        # Objectness
        # tobj[img_id, anchor_index, cy_index, cx_index] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)
        for x in range(len(img_id)):
            tobj[img_id[x], anchor_index[x], cy_index[x], cx_index[x]] = (1.0 - self.gr) + self.gr * iou[x].detach().clamp(0).type(tobj.dtype)
        lobj = self.BCEobj(p[..., 4], tobj) * balance
        balance = balance * 0.9999 + 0.0001 / lobj.detach().item() if self.autobalance else balance
        # Classification
        if self.class_num <= 1:
            return lbox, 0, lobj
        t = torch.full_like(ps[:, 5:], self.cn, device=self.device)
        t[range(target_num), tcls] = self.cp
        lcls = self.BCEcls(ps[:, 5:], t)
        return lbox, lcls, lobj

    def __call__(self, pred, target):
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_target(pred, target)
        # pred: [(n, c, h1, w1, 9), (n, c, h2, w2, 9), (n, c, h3, w3, 9)]
        for i, p in enumerate(pred):
            libox, licls, liobj = self.compute_loss(p, tcls[i], tbox[i], indices[i], anchors[i], self.balance[i])
            lbox += libox
            lcls += licls
            lobj += liobj
        self.balance = [x / self.balance[self.ssi] for x in self.balance] if self.autobalance else self.balance
        lbox *= self.box
        lobj *= self.obj
        lcls *= self.cls
        bs = pred[0].size(0)
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True, GIou=False, DIou=False, CIou=False, eps=1e-9):
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
        coords = (b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        whs = (w1, h1, w2, h2)
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        if not GIou and not DIou and not CIou:
            return iou
        elif GIou:
            return Yolov5Loss.compute_GIou(iou, coords, union)[-1]
        elif DIou:
            return Yolov5Loss.compute_DIou(iou, coords, union)[-1]
        elif CIou:
            return Yolov5Loss.compute_CIou(iou, coords, whs, union)

    @staticmethod
    def compute_GIou(iou, coords, union, eps=1e-9):
        b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2 = coords
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c_area = cw * ch + eps
        giou = iou - (c_area - union) / c_area
        return cw, ch, giou

    @staticmethod
    def compute_DIou(iou, coords, union, eps=1e-9):
        cw, ch, _ = Yolov5Loss.compute_GIou(iou, coords, union)
        c2 = cw ** 2 + ch ** 2 + eps
        b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2 = coords
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        diou = iou - rho2 / c2
        return c2, rho2, diou

    @staticmethod
    def compute_CIou(iou, coords, whs, union, eps=1e-9):
        cw, ch, _ = Yolov5Loss.compute_GIou(iou, coords, union)
        c2, rho2, _ = Yolov5Loss.compute_DIou(iou, coords, union)
        w1, h1, w2, h2 = whs
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / ((1 + eps) - iou + v)
        ciou = iou - (rho2 / c2 + v * alpha)
        return ciou

    @staticmethod
    def smoothBCE(eps=0.1):
        return 1.0 - 0.5 * eps, 0.5 * eps


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.device = device

    def _match(self, threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
        overlaps = self._jaccard(
            truths,
            self._pointForm(priors)
        )
        # (Bipartite Matching)
        # [1,num_objects] best prior for each ground truth
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

        # ignore hard gt
        valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
        best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
        if best_prior_idx_filter.shape[0] <= 0:
            loc_t[idx] = 0
            conf_t[idx] = 0
            return

        # [1,num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)
        best_prior_idx.squeeze_(1)
        best_prior_idx_filter.squeeze_(1)
        best_prior_overlap.squeeze_(1)
        best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior
        # ensure every gt matches with its prior of max overlap
        for j in range(best_prior_idx.size(0)):
            best_truth_idx[best_prior_idx[j]] = j

        matches = truths[best_truth_idx]
        conf = labels[best_truth_idx]
        conf[best_truth_overlap < threshold] = 0
        loc = self._encode(matches, priors, variances)

        matches_landm = landms[best_truth_idx]
        landm = self._encodeLandm(matches_landm, priors, variances)
        loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
        conf_t[idx] = conf  # [num_priors] top class label for each prior
        landm_t[idx] = landm

    def _jaccard(self, box_a, box_b):
        inter = self._intersect(box_a, box_b)
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
                (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) *
                (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def _pointForm(self, boxes):
        return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

    def _intersect(self, box_a, box_b):
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def _encode(self, matched, priors, variances):
        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

    def _encodeLandm(self, matched, priors, variances):
        # dist b/t match center and prior's center
        matched = torch.reshape(matched, (matched.size(0), 5, 2))
        priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
        priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
        priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
        priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
        priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
        g_cxcy = matched[:, :, :2] - priors[:, :, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, :, 2:])
        # g_cxcy /= priors[:, :, 2:]
        g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
        # return target for smooth_l1_loss
        return g_cxcy

    def _logSumExp(self, x):
        x_max = x.data.max()
        return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

    def forward(self, predictions, priors, targets):
        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            self._match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)

        loc_t = loc_t.to(self.device)
        conf_t = conf_t.to(self.device)
        landm_t = landm_t.to(self.device)

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = self._logSumExp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm
