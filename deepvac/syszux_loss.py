import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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