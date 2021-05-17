import numpy as np
import random
from .base_aug import AugBase
from .perspective_helper import apply_perspective_transform
from .remaper_helper import Remaper
from .line_helper import Liner
from .emboss_helper import apply_emboss

class TextRendererPerspectiveAug(AugBase):
    def __init__(self, deepvac_config):
        super(TextRendererPerspectiveAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.text_renderer_perspective_max_x = addUserConfig('text_renderer_perspective_max_x', self.config.text_renderer_perspective_max_x, 10)
        self.config.text_renderer_perspective_max_y = addUserConfig('text_renderer_perspective_max_y', self.config.text_renderer_perspective_max_y, 10)
        self.config.text_renderer_perspective_max_z = addUserConfig('text_renderer_perspective_max_z', self.config.text_renderer_perspective_max_z, 5)

    def __call__(self, img):
        self.auditInput(img)
        return apply_perspective_transform(img, self.config.text_renderer_perspective_max_x, self.config.text_renderer_perspective_max_y, self.config.text_renderer_perspective_max_z)

class TextRendererCurveAug(AugBase):
    def __init__(self, deepvac_config):
        super(TextRendererCurveAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        self.auditInput(img)
        h, w = img.shape[:2]
        re_img, text_box_pnts = Remaper().apply(img, [[0,0],[w,0],[w,h],[0,h]])
        return re_img

class TextRendererLineAug(AugBase):
    def __init__(self, deepvac_config):
        super(TextRendererLineAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.text_renderer_line_offset = addUserConfig('text_renderer_line_offset', self.config.text_renderer_line_offset, 5)

    def __call__(self, img):
        self.auditInput(img)
        h, w = img.shape[:2]
        offset = self.config.text_renderer_line_offset
        pos = [[offset,offset],[w-offset,offset],[w-offset,h-offset],[offset,h-offset]]
        re_img, text_box_pnts = Liner().apply(img, pos)
        return re_img

class TextRendererEmbossAug(AugBase):
    def __init__(self, deepvac_config):
        super(TextRendererEmbossAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        self.auditInput(img)
        return apply_emboss(img)

class TextRendererReverseAug(AugBase):
    def __init__(self, deepvac_config):
        super(TextRendererReverseAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        self.auditInput(img)
        offset = np.random.randint(-10, 10)
        return 255 + offset - img