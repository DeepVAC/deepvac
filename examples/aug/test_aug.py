import sys
sys.path.append("..")
import os
import cv2
from syszux_aug import *
from conf import *

speckle = SpeckleAug(config.text)
affine = AffineAug(config.text)
perspect = PerspectAug(config.text)
gaussian = GaussianAug(config.text)
horline = HorlineAug(config.text)
verline = VerlineAug(config.text)
lrmotion = LRmotionAug(config.text)
udmotion = UDmotionAug(config.text)
noisy = NoisyAug(config.text)
img_names = os.listdir('/gemfield/hostpv/gemfield/pure')
for img_name in img_names:
    img = cv2.imread('/gemfield/hostpv/gemfield/pure/'+img_name)
    img_speckle = speckle(img)
    img_affine = affine(img)
    img_perspect = perspect(img)
    img_gaussian = gaussian(img)
    img_horline = horline(img)
    img_verline = verline(img)
    img_lrmotion = lrmotion(img)
    img_udmotion = udmotion(img)
    img_noisy = noisy(img)
    cv2.imwrite('/gemfield/hostpv/gemfield/aug_pure/speckle_'+img_name,img_speckle)
    cv2.imwrite('/gemfield/hostpv/gemfield/aug_pure/affine_'+img_name,img_affine)
    cv2.imwrite('/gemfield/hostpv/gemfield/aug_pure/perspect_'+img_name,img_perspect)
    cv2.imwrite('/gemfield/hostpv/gemfield/aug_pure/gaussian_'+img_name,img_gaussian)
    cv2.imwrite('/gemfield/hostpv/gemfield/aug_pure/horline_'+img_name,img_horline)
    cv2.imwrite('/gemfield/hostpv/gemfield/aug_pure/verline_'+img_name,img_verline)
    cv2.imwrite('/gemfield/hostpv/gemfield/aug_pure/lrmotion_'+img_name,img_lrmotion)
    cv2.imwrite('/gemfield/hostpv/gemfield/aug_pure/udmotion_'+img_name,img_udmotion)
    cv2.imwrite('/gemfield/hostpv/gemfield/aug_pure/noisy_'+img_name,img_noisy)
