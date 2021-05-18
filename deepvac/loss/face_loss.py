import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import LossBase

class MultiBoxLoss(LossBase):
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
    def __init__(self, embedding_size, class_num, world_size=1, s=64.0, m=0.50):
        super(CurricularFace, self).__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.world_size = world_size
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
            if self.world_size != 1:
                dist.all_reduce(self.t, dist.ReduceOp.SUM)
            self.t = self.t / self.world_size
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output