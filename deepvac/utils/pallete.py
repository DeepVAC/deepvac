import numpy as np
import cv2

pallete20 = [[255, 255, 255],
            [128, 64,  128],
            [244, 35,  232],
            [70,  70,  70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70,  130, 180],
            [220, 20,  60],
            [255, 0,   0],
            [0,   0,   142],
            [0,   0,   70],
            [0,   60,  100],
            [0,   80,  100],
            [0,   0,   230],
            [119, 11,  32]]

def getOverlayFromSegMask(img, mask):
    assert len(mask.shape) == 2, 'only support 2-D mask, got {}'.format(len(mask.shape))
    assert img.shape[:2] == mask.shape[:2], 'img shape not equal mask shape: {} vs {}'.format(img.shape[:2], mask.shape[:2])
    class_map_np_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for idx in np.unique(mask):
        [r, g, b] = pallete20[idx]
        class_map_np_color[mask == idx] = [b, g, r]
    overlayed = cv2.addWeighted(img, 0.5, class_map_np_color, 0.5, 0)
    return overlayed