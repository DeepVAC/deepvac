# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank
import time
import cv2

class AlignFace(object):
    def __init__(self):
        # reference facial points, a list of coordinates (x,y)
        self.REFERENCE_FACIAL_POINTS_96x112 = [
            [30.29459953, 51.69630051],
            [65.53179932, 51.50139999],
            [48.02519989, 71.73660278],
            [33.54930115, 92.3655014],
            [62.72990036, 92.20410156]
        ]
        # made by gemfield
        self.REFERENCE_FACIAL_POINTS_112x112 = [
            [38.29459953, 51.69630051],
            [73.53179932, 51.50139999],
            [56.02519989, 71.73660278],
            [41.54930115, 92.3655014 ],
            [70.72990036, 92.20410156]
        ]

    def __call__(self, frame, facial_5pts):
        # shape from (10,) to (2, 5)
        # x1,x2...y1,y2... or  x1,y1,x2,y2...
        facial = []
        x = facial_5pts[::2]
        y = facial_5pts[1::2]
        facial.append(x)
        facial.append(y)
        dst_img = self.warpAndCrop(frame, facial, (112, 112))
        return dst_img

    def warpAndCrop(self, src_img, facial_pts, crop_size):
        reference_pts = self.REFERENCE_FACIAL_POINTS_112x112
        ref_pts = np.float32(reference_pts)
        ref_pts_shp = ref_pts.shape

        if ref_pts_shp[0] == 2:
            ref_pts = ref_pts.T

        src_pts = np.float32(facial_pts)
        src_pts_shp = src_pts.shape
        if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
            raise Exception('facial_pts.shape must be (K,2) or (2,K) and K>2')
        # 2*5 to 5*2
        if src_pts_shp[0] == 2:
            src_pts = src_pts.T

        if src_pts.shape != ref_pts.shape:
            raise Exception('facial_pts and reference_pts must have the same shape: {} vs {}'.format(src_pts.shape, ref_pts.shape) )

        tfm = self.getAffineTransform(src_pts, ref_pts)

        face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))
        return face_img

    def getAffineTransform(self, uv, xy):
        options = {'K': 2}
        # Solve for trans1
        trans1, trans1_inv = self.findNonreflectiveSimilarity(uv, xy, options)
        # manually reflect the xy data across the Y-axis
        xyR = xy
        xyR[:, 0] = -1 * xyR[:, 0]

        trans2r, trans2r_inv = self.findNonreflectiveSimilarity(uv, xyR, options)

        # manually reflect the tform to undo the reflection done on xyR
        TreflectY = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        trans2 = np.dot(trans2r, TreflectY)

        # Figure out if trans1 or trans2 is better
        xy1 = self.tformfwd(trans1, uv)
        norm1 = norm(xy1 - xy)

        xy2 = self.tformfwd(trans2, uv)
        norm2 = norm(xy2 - xy)

        if norm1 <= norm2:
            trans = trans1
        else:
            trans2_inv = inv(trans2)
            trans = trans2

        cv2_trans = trans[:, 0:2].T
        return cv2_trans

    def findNonreflectiveSimilarity(self, uv, xy, options=None):
        options = {'K': 2}

        K = options['K']
        M = xy.shape[0]
        x = xy[:, 0].reshape((-1, 1))
        y = xy[:, 1].reshape((-1, 1))

        tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
        tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
        X = np.vstack((tmp1, tmp2))

        u = uv[:, 0].reshape((-1, 1))
        v = uv[:, 1].reshape((-1, 1))
        U = np.vstack((u, v))

        if rank(X) >= 2 * K:
            r, _, _, _ = lstsq(X, U, rcond=-1)
            r = np.squeeze(r)
        else:
            raise Exception('cp2tform:twoUniquePointsReq')
        sc = r[0]
        ss = r[1]
        tx = r[2]
        ty = r[3]

        Tinv = np.array([
            [sc, -ss, 0],
            [ss, sc, 0],
            [tx, ty, 1]
        ])

        T = inv(Tinv)
        T[:, 2] = np.array([0, 0, 1])
        return T, Tinv

    def tformfwd(self, trans, uv):
        uv = np.hstack((
            uv, np.ones((uv.shape[0], 1))
        ))
        xy = np.dot(uv, trans)
        xy = xy[:, 0:-1]
        return xy

    def drawBoxes(self, img, imgname):
        write_path = './results/{}/bbox/'.format( time.strftime("%Y%m%d", time.localtime()) )
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        cv2.imwrite('{}{}_aligned.jpg'.format(write_path, imgname), img)
