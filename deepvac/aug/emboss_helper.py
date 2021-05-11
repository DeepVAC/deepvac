import numpy as np

emboss_kernal = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
])

def apply_emboss(word_img):
    return cv2.filter2D(word_img, -1, emboss_kernal)