# -*- coding:utf-8 -*-
# Author: RubanSeven
import math
import cv2
import numpy as np
import random

from PIL import Image
from deepvac.syszux_log import LOG
from functools import reduce

class WarpMLS:
    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    @staticmethod
    def __bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x

    def generate(self):
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        w = np.zeros(self.pt_count, dtype=np.float32)

        if self.pt_count < 2:
            return

        i = 0
        while 1:
            if self.dst_w <= i < self.dst_w + self.grid_size - 1:
                i = self.dst_w - 1
            elif i >= self.dst_w:
                break

            j = 0
            while 1:
                if self.dst_h <= j < self.dst_h + self.grid_size - 1:
                    j = self.dst_h - 1
                elif j >= self.dst_h:
                    break

                sw = 0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(self.pt_count):
                    if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                        break

                    w[k] = 1. / ((i - self.dst_pts[k][0]) * (i - self.dst_pts[k][0]) +
                                 (j - self.dst_pts[k][1]) * (j - self.dst_pts[k][1]))

                    sw += w[k]
                    swp = swp + w[k] * np.array(self.dst_pts[k])
                    swq = swq + w[k] * np.array(self.src_pts[k])

                if k == self.pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    miu_s = 0
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)

                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue

                        pt_i = self.dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])

                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] = np.sum(pt_i * cur_pt) * self.src_pts[k][0] - \
                                    np.sum(pt_j * cur_pt) * self.src_pts[k][1]
                        tmp_pt[1] = -np.sum(pt_i * cur_pt_j) * self.src_pts[k][0] + \
                                    np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
                        tmp_pt *= (w[k] / miu_s)
                        new_pt += tmp_pt

                    new_pt += qstar
                else:
                    new_pt = self.src_pts[k]

                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j

                j += self.grid_size
            i += self.grid_size

    def gen_img(self):
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)

        for i in np.arange(0, self.dst_h, self.grid_size):
            for j in np.arange(0, self.dst_w, self.grid_size):
                ni = i + self.grid_size
                nj = j + self.grid_size
                w = h = self.grid_size
                if ni >= self.dst_h:
                    ni = self.dst_h - 1
                    h = ni - i + 1
                if nj >= self.dst_w:
                    nj = self.dst_w - 1
                    w = nj - j + 1

                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self.__bilinear_interp(di / h, dj / w,
                                                 self.rdx[i, j], self.rdx[i, nj],
                                                 self.rdx[ni, j], self.rdx[ni, nj])
                delta_y = self.__bilinear_interp(di / h, dj / w,
                                                 self.rdy[i, j], self.rdy[i, nj],
                                                 self.rdy[ni, j], self.rdy[ni, nj])
                nx = j + dj + delta_x * self.trans_ratio
                ny = i + di + delta_y * self.trans_ratio
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)

                if len(self.src.shape) == 3:
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i:i + h, j:j + w] = self.__bilinear_interp(x,
                                                               y,
                                                               self.src[nyi, nxi],
                                                               self.src[nyi, nxi1],
                                                               self.src[nyi1, nxi],
                                                               self.src[nyi1, nxi1]
                                                               )

        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.uint8)

        return dst

def separateBN4OptimizerPG(modules):
    paras_only_bn = []
    paras_wo_bn = []
    memo = set()
    gemfield_set = set()
    gemfield_set.update(set(modules.parameters()))
    LOG.logI("separateBN4OptimizerPG set len: {}".format(len(gemfield_set)))
    named_modules = modules.named_modules(prefix='')
    for module_prefix, module in named_modules:
        if "module" not in module_prefix:
            LOG.logI("separateBN4OptimizerPG skip {}".format(module_prefix))
            continue

        members = module._parameters.items()
        for k, v in members:
            name = module_prefix + ('.' if module_prefix else '') + k
            if v is None:
                continue
            if v in memo:
                continue
            memo.add(v)
            if "batchnorm" in str(module.__class__):
                paras_only_bn.append(v)
            else:
                paras_wo_bn.append(v)

    LOG.logI("separateBN4OptimizerPG param len: {} - {}".format(len(paras_wo_bn),len(paras_only_bn)))
    return paras_only_bn, paras_wo_bn

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_image(image_path):
    # if the image_path is a remote url, read the image at first
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path

    # convert non-RGB mode to RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def get_thumbnail(image):
    image.thumbnail((256, 256))
    return image

def get_colors(image_path):
    """ image instance
    """
    image = get_image(image_path)

    """ image thumbnail
        size: 256 * 256
        reduce the calculate time 
    """
    thumbnail = get_thumbnail(image)


    """ calculate the max colors the image cound have
        if the color is different in every pixel, the color counts may be the max.
        so : 
        max_colors = image.height * image.width
    """
    image_height = thumbnail.height
    image_width = thumbnail.width
    max_colors = image_height * image_width

    image_colors = image.getcolors(max_colors)
    return image_colors


def sort_by_rgb(colors_tuple):
    """ colors_tuple contains color count and color RGB
        we want to sort the tuple by RGB
        tuple[1]
    """
    sorted_tuple = sorted(colors_tuple, key=lambda x:x[1])
    return sorted_tuple

def rgb_maximum(colors_tuple):
    """ 
        colors_r max min
        colors_g max min
        colors_b max min

    """
    r_sorted_tuple = sorted(colors_tuple, key=lambda x:x[1][0])
    g_sorted_tuple = sorted(colors_tuple, key=lambda x:x[1][1])
    b_sorted_tuple = sorted(colors_tuple, key=lambda x:x[1][2])

    r_min = r_sorted_tuple[0][1][0]
    g_min = g_sorted_tuple[0][1][1]
    b_min = b_sorted_tuple[0][1][2]

    r_max = r_sorted_tuple[len(colors_tuple)-1][1][0]
    g_max = g_sorted_tuple[len(colors_tuple)-1][1][1]
    b_max = b_sorted_tuple[len(colors_tuple)-1][1][2]

    return {
        "r_max":r_max,
        "r_min":r_min,
        "g_max":g_max,
        "g_min":g_min,
        "b_max":b_max,
        "b_min":b_min,
        "r_dvalue":(r_max-r_min)/3,
        "g_dvalue":(g_max-g_min)/3,
        "b_dvalue":(b_max-b_min)/3
    }

def group_by_accuracy(sorted_tuple, accuracy=3):
    """ group the colors by the accuaracy was given
        the R G B colors will be depart to accuracy parts
        default accuracy = 3
        d_value = (max-min)/3
        [min, min+d_value), [min+d_value, min+d_value*2), [min+d_value*2, max)
    """
    rgb_maximum_json = rgb_maximum(sorted_tuple)
    r_min = rgb_maximum_json["r_min"]
    g_min = rgb_maximum_json["g_min"]
    b_min = rgb_maximum_json["b_min"]
    r_dvalue = rgb_maximum_json["r_dvalue"]
    g_dvalue = rgb_maximum_json["g_dvalue"]
    b_dvalue = rgb_maximum_json["b_dvalue"]

    rgb = [
            [[[], [], []], [[], [], []], [[], [], []]],
            [[[], [], []], [[], [], []], [[], [], []]],
            [[[], [], []], [[], [], []], [[], [], []]]
        ]

    for color_tuple in sorted_tuple:
        r_tmp_i = color_tuple[1][0]
        g_tmp_i = color_tuple[1][1]
        b_tmp_i = color_tuple[1][2]
        r_idx = 0 if r_tmp_i < (r_min+r_dvalue) else 1 if r_tmp_i < (r_min+r_dvalue*2) else 2
        g_idx = 0 if g_tmp_i < (g_min+g_dvalue) else 1 if g_tmp_i < (g_min+g_dvalue*2) else 2
        b_idx = 0 if b_tmp_i < (b_min+b_dvalue) else 1 if b_tmp_i < (b_min+b_dvalue*2) else 2
        rgb[r_idx][g_idx][b_idx].append(color_tuple)

    return rgb


def get_weighted_mean(grouped_image_color):
    """ calculate every group's weighted mean

        r_weighted_mean = sigma(r * count) / sigma(count)
        g_weighted_mean = sigma(g * count) / sigma(count)
        b_weighted_mean = sigma(b * count) / sigma(count)
    """
    sigma_count = 0
    sigma_r = 0
    sigma_g = 0
    sigma_b = 0

    for item in grouped_image_color:
        sigma_count += item[0]
        sigma_r += item[1][0] * item[0]
        sigma_g += item[1][1] * item[0]
        sigma_b += item[1][2] * item[0]

    r_weighted_mean = int(sigma_r / sigma_count)
    g_weighted_mean = int(sigma_g / sigma_count)
    b_weighted_mean = int(sigma_b / sigma_count)
    
    weighted_mean = (sigma_count, (r_weighted_mean, g_weighted_mean, b_weighted_mean))
    return weighted_mean

class Haishoku(object):

    """ init Haishoku obj
    """
    def __init__(self):
        self.dominant = None
        self.palette = None

    """ immediate api

        1. showPalette
        2. showDominant
        3. getDominant
        4. getPalette
    """
    def getColorsMean(image):
        # get colors tuple 
        image_colors = get_colors(image)

        # sort the image colors tuple
        sorted_image_colors = sort_by_rgb(image_colors)

        # group the colors by the accuaracy
        grouped_image_colors = group_by_accuracy(sorted_image_colors)

        # get the weighted mean of all colors
        colors_mean = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    grouped_image_color = grouped_image_colors[i][j][k]
                    if 0 != len(grouped_image_color):
                        color_mean = get_weighted_mean(grouped_image_color)
                        colors_mean.append(color_mean)

        # return the most 8 colors
        temp_sorted_colors_mean = sorted(colors_mean)
        if 8 < len(temp_sorted_colors_mean):
            colors_mean = temp_sorted_colors_mean[len(temp_sorted_colors_mean)-8 : len(temp_sorted_colors_mean)]
        else:
            colors_mean = temp_sorted_colors_mean

        # sort the colors_mean
        colors_mean = sorted(colors_mean, reverse=True)

        return colors_mean
        
    def getDominant(image=None):
        # get the colors_mean
        colors_mean = Haishoku.getColorsMean(image)
        colors_mean = sorted(colors_mean, reverse=True)

        # get the dominant color
        dominant_tuple = colors_mean[0]
        dominant = dominant_tuple[1]
        return dominant

emboss_kernal = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
])


def apply_emboss(word_img):
    return cv2.filter2D(word_img, -1, emboss_kernal)

class LineState(object):
    tableline_x_offsets = range(1, 5)
    tableline_y_offsets = range(1, 5)
    tableline_thickness = [2, 3]

    # 0/1/2/3: 仅单边（左上右下）
    # 4/5/6/7: 两边都有线（左上，右上，右下，左下）
    tableline_options = range(0, 8)

    middleline_thickness = [1, 2, 3]
    middleline_thickness_p = [0.2, 0.7, 0.1]


class Liner(object):
    def __init__(self):
        self.linestate = LineState()
        self.cfg = {}
        self.cfg['line'] = {}

        self.cfg['line']['under_line'] = {}
        self.cfg['line']['under_line']['enable'] = True
        self.cfg['line']['under_line']['fraction'] = 0.5

        self.cfg['line']['table_line'] = {}
        self.cfg['line']['table_line']['enable'] = True
        self.cfg['line']['table_line']['fraction'] = 0.5

        self.cfg['line']['middle_line'] = {}
        self.cfg['line']['middle_line']['enable'] = True
        self.cfg['line']['middle_line']['fraction'] = 0


        self.cfg['line_color'] = {}
        self.cfg['line_color']['enable'] = True

        self.cfg['line_color']['black'] = {}
        self.cfg['line_color']['black']['fraction'] = 0.5
        self.cfg['line_color']['black']['l_boundary'] = [0,0,0]
        self.cfg['line_color']['black']['h_boundary'] = [64,64,64]

        self.cfg['line_color']['blue'] = {}
        self.cfg['line_color']['blue']['fraction'] = 0.5
        self.cfg['line_color']['blue']['l_boundary'] = [0,0,150]
        self.cfg['line_color']['blue']['h_boundary'] = [60,64,255]


    def get_line_color(self):
        p = []
        colors = []
        for k, v in self.cfg['line_color'].items():
            if k == 'enable':
                continue
            p.append(v['fraction'])
            colors.append(k)

        # pick color by fraction
        color_name = np.random.choice(colors, p=p)
        l_boundary = self.cfg['line_color'][color_name]['l_boundary']
        h_boundary = self.cfg['line_color'][color_name]['h_boundary']
        # random color by low and high RGB boundary
        r = np.random.randint(l_boundary[0], h_boundary[0])
        g = np.random.randint(l_boundary[1], h_boundary[1])
        b = np.random.randint(l_boundary[2], h_boundary[2])
        return b, g, r

    def apply(self, word_img, text_box_pnts):
        """
        :param word_img:  word image with big background
        :param text_box_pnts: left-top, right-top, right-bottom, left-bottom of text word
        :return:
        """
        line_p = []
        funcs = []

        if self.cfg['line']['under_line']['enable']:
            line_p.append(self.cfg['line']['under_line']['fraction'])
            funcs.append(self.apply_under_line)

        if self.cfg['line']['table_line']['enable']:
            line_p.append(self.cfg['line']['table_line']['fraction'])
            funcs.append(self.apply_table_line)

        if self.cfg['line']['middle_line']['enable']:
            line_p.append(self.cfg['line']['middle_line']['fraction'])
            funcs.append(self.apply_middle_line)


        if len(line_p) == 0:
            return word_img, text_box_pnts

        line_effect_func = np.random.choice(funcs, p=line_p)

        if self.cfg['line_color']['enable'] or self.cfg['font_color']['enable']:
            line_color = self.get_line_color()
        else:
            line_color = word_color + random.randint(0, 10)

        return line_effect_func(word_img, text_box_pnts, line_color)

    def apply_under_line(self, word_img, text_box_pnts, line_color):
        y_offset = random.choice([0, 1])

        text_box_pnts[2][1] += y_offset
        text_box_pnts[3][1] += y_offset

        dst = cv2.line(word_img,
                       (text_box_pnts[2][0], text_box_pnts[2][1]),
                       (text_box_pnts[3][0], text_box_pnts[3][1]),
                       color=line_color,
                       thickness=3,
                       lineType=cv2.LINE_AA)

        return dst, text_box_pnts

    def apply_table_line(self, word_img, text_box_pnts, line_color):
        """
        共有 8 种可能的画法，横线横穿整张 word_img
        0/1/2/3: 仅单边（左上右下）
        4/5/6/7: 两边都有线（左上，右上，右下，左下）
        """
        dst = word_img
        option = random.choice(self.linestate.tableline_options)
        thickness = random.choice(self.linestate.tableline_thickness)

        top_y_offset = random.choice(self.linestate.tableline_y_offsets)
        bottom_y_offset = random.choice(self.linestate.tableline_y_offsets)
        left_x_offset = random.choice(self.linestate.tableline_x_offsets)
        right_x_offset = random.choice(self.linestate.tableline_x_offsets)

        def is_top():
            return option in [1, 4, 5]

        def is_bottom():
            return option in [3, 6, 7]

        def is_left():
            return option in [0, 4, 7]

        def is_right():
            return option in [2, 5, 6]

        if is_top():
            text_box_pnts[0][1] -= top_y_offset
            text_box_pnts[1][1] -= top_y_offset

        if is_bottom():
            text_box_pnts[2][1] += bottom_y_offset
            text_box_pnts[3][1] += bottom_y_offset

        if is_left():
            text_box_pnts[0][0] -= left_x_offset
            text_box_pnts[3][0] -= left_x_offset

        if is_right():
            text_box_pnts[1][0] += right_x_offset
            text_box_pnts[2][0] += right_x_offset

        if is_bottom():
            dst = cv2.line(dst,
                           (0, text_box_pnts[2][1]),
                           (word_img.shape[1], text_box_pnts[3][1]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_top():
            dst = cv2.line(dst,
                           (0, text_box_pnts[0][1]),
                           (word_img.shape[1], text_box_pnts[1][1]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_left():
            dst = cv2.line(dst,
                           (text_box_pnts[0][0], 0),
                           (text_box_pnts[3][0], word_img.shape[0]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_right():
            dst = cv2.line(dst,
                           (text_box_pnts[1][0], 0),
                           (text_box_pnts[2][0], word_img.shape[0]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        return dst, text_box_pnts

    def apply_middle_line(self, word_img, text_box_pnts, line_color):
        y_center = int((text_box_pnts[0][1] + text_box_pnts[3][1]) / 2)

        thickness = np.random.choice(self.linestate.middleline_thickness, p=self.linestate.middleline_thickness_p)

        dst = cv2.line(word_img,
                       (text_box_pnts[0][0], y_center),
                       (text_box_pnts[1][0], y_center),
                       color=line_color,
                       thickness=thickness,
                       lineType=cv2.LINE_AA)

        return dst, text_box_pnts

# http://planning.cs.uiuc.edu/node102.html
def get_rotate_matrix(x, y, z):
    """
    按照 zyx 的顺序旋转，输入角度单位为 degrees, 均为顺时针旋转
    :param x: X-axis
    :param y: Y-axis
    :param z: Z-axis
    :return:
    """
    x = math.radians(x)
    y = math.radians(y)
    z = math.radians(z)

    c, s = math.cos(y), math.sin(y)
    M_y = np.matrix([[c, 0., s, 0.],
                     [0., 1., 0., 0.],
                     [-s, 0., c, 0.],
                     [0., 0., 0., 1.]])

    c, s = math.cos(x), math.sin(x)
    M_x = np.matrix([[1., 0., 0., 0.],
                     [0., c, -s, 0.],
                     [0., s, c, 0.],
                     [0., 0., 0., 1.]])

    c, s = math.cos(z), math.sin(z)
    M_z = np.matrix([[c, -s, 0., 0.],
                     [s, c, 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])

    return M_x * M_y * M_z


def cliped_rand_norm(mu=0, sigma3=1):
    """
    :param mu: 均值
    :param sigma3: 3 倍标准差， 99% 的数据落在 (mu-3*sigma, mu+3*sigma)
    :return:
    """
    # 标准差
    sigma = sigma3 / 3
    dst = sigma * np.random.randn() + mu
    dst = np.clip(dst, 0 - sigma3, sigma3)
    return dst


def warpPerspective(src, M33, sl, gpu):
    if gpu:
        from libs.gpu.GpuWrapper import cudaWarpPerspectiveWrapper
        dst = cudaWarpPerspectiveWrapper(src.astype(np.uint8), M33, (sl, sl), cv2.INTER_CUBIC)
    else:
        dst = cv2.warpPerspective(src, M33, (sl, sl), flags=cv2.INTER_CUBIC)
    return dst


# https://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles
# https://nbviewer.jupyter.org/github/manisoftwartist/perspectiveproj/blob/master/perspective.ipynb
# http://planning.cs.uiuc.edu/node102.html
class PerspectiveTransform(object):
    def __init__(self, x, y, z, scale, fovy):
        self.x = x
        self.y = y
        self.z = z
        self.scale = scale
        self.fovy = fovy

    def transform_image(self, src, gpu=False):
        if len(src.shape) > 2:
            H, W, C = src.shape
        else:
            H, W = src.shape

        M33, sl, _, ptsOut = self.get_warp_matrix(W, H, self.x, self.y, self.z, self.scale, self.fovy)
        sl = int(sl)

        dst = warpPerspective(src, M33, sl, gpu)

        return dst, M33, ptsOut

    def transform_pnts(self, pnts, M33):
        """
        :param pnts: 2D pnts, left-top, right-top, right-bottom, left-bottom
        :param M33: output from transform_image()
        :return: 2D pnts apply perspective transform
        """
        pnts = np.asarray(pnts, dtype=np.float32)
        pnts = np.array([pnts])
        dst_pnts = cv2.perspectiveTransform(pnts, M33)[0]

        return dst_pnts

    def get_warped_pnts(self, ptsIn, ptsOut, W, H, sidelength):
        ptsIn2D = ptsIn[0, :]
        ptsOut2D = ptsOut[0, :]
        ptsOut2Dlist = []
        ptsIn2Dlist = []

        for i in range(0, 4):
            ptsOut2Dlist.append([ptsOut2D[i, 0], ptsOut2D[i, 1]])
            ptsIn2Dlist.append([ptsIn2D[i, 0], ptsIn2D[i, 1]])

        pin = np.array(ptsIn2Dlist) + [W / 2., H / 2.]
        pout = (np.array(ptsOut2Dlist) + [1., 1.]) * (0.5 * sidelength)
        pin = pin.astype(np.float32)
        pout = pout.astype(np.float32)

        return pin, pout

    def get_warp_matrix(self, W, H, x, y, z, scale, fV):
        fVhalf = np.deg2rad(fV / 2.)
        d = np.sqrt(W * W + H * H)
        sideLength = scale * d / np.cos(fVhalf)
        h = d / (2.0 * np.sin(fVhalf))
        n = h - (d / 2.0)
        f = h + (d / 2.0)

        # Translation along Z-axis by -h
        T = np.eye(4, 4)
        T[2, 3] = -h

        # Rotation matrices around x,y,z
        R = get_rotate_matrix(x, y, z)

        # Projection Matrix
        P = np.eye(4, 4)
        P[0, 0] = 1.0 / np.tan(fVhalf)
        P[1, 1] = P[0, 0]
        P[2, 2] = -(f + n) / (f - n)
        P[2, 3] = -(2.0 * f * n) / (f - n)
        P[3, 2] = -1.0

        # pythonic matrix multiplication
        M44 = reduce(lambda x, y: np.matmul(x, y), [P, T, R])

        # shape should be 1,4,3 for ptsIn and ptsOut since perspectiveTransform() expects data in this way.
        # In C++, this can be achieved by Mat ptsIn(1,4,CV_64FC3);
        ptsIn = np.array([[
            [-W / 2., H / 2., 0.],
            [W / 2., H / 2., 0.],
            [W / 2., -H / 2., 0.],
            [-W / 2., -H / 2., 0.]
        ]])
        ptsOut = cv2.perspectiveTransform(ptsIn, M44)

        ptsInPt2f, ptsOutPt2f = self.get_warped_pnts(ptsIn, ptsOut, W, H, sideLength)

        # check float32 otherwise OpenCV throws an error
        assert (ptsInPt2f.dtype == np.float32)
        assert (ptsOutPt2f.dtype == np.float32)
        M33 = cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f).astype(np.float32)

        return M33, sideLength, ptsInPt2f, ptsOutPt2f

def apply_perspective_transform(img, max_x, max_y, max_z):
    x = cliped_rand_norm(0, max_x)
    y = cliped_rand_norm(0, max_y)
    z = cliped_rand_norm(0, max_z)

    transformer = PerspectiveTransform(x, y, z, scale=1.0, fovy=50)
    dst_img, M33, dst_img_pnts = transformer.transform_image(img, False)
    dst_img_pnts = np.array(dst_img_pnts,dtype = np.int32)

    x_min, y_min = np.min(dst_img_pnts,axis=0)
    x_max, y_max = np.max(dst_img_pnts,axis=0)
    return dst_img[y_min:y_max, x_min:x_max]

class Remaper(object):
    def __init__(self):
        self.period = 360  
        self.min = 1
        self.max = 5

    def apply(self, word_img, text_box_pnts):
        """
        :param word_img:  word image with big background
        :param text_box_pnts: left-top, right-top, right-bottom, left-bottom of text word
        :return:
        """
        max_val = np.random.uniform(self.min, self.max)

        h = word_img.shape[0]
        w = word_img.shape[1]

        img_x = np.zeros((h, w), np.float32)
        img_y = np.zeros((h, w), np.float32)

        xmin = text_box_pnts[0][0]
        xmax = text_box_pnts[1][0]
        ymin = text_box_pnts[0][1]
        ymax = text_box_pnts[2][1]

        remap_y_min = ymin
        remap_y_max = ymax

        for y in range(h):
            for x in range(w):
                remaped_y = y + self._remap_y(x, max_val)

                if y == ymin:
                    if remaped_y < remap_y_min:
                        remap_y_min = remaped_y

                if y == ymax:
                    if remaped_y > remap_y_max:
                        remap_y_max = remaped_y

                # 某一个位置的 y 值应该为哪个位置的 y 值
                img_y[y, x] = remaped_y
                # 某一个位置的 x 值应该为哪个位置的 x 值
                img_x[y, x] = x

        remaped_text_box_pnts = [
            [xmin, remap_y_min],
            [xmax, remap_y_min],
            [xmax, remap_y_max],
            [xmin, remap_y_max]
        ]

        # TODO: use cuda::remap
        dst = cv2.remap(word_img, img_x, img_y, cv2.INTER_CUBIC)
        return dst, remaped_text_box_pnts

    def _remap_y(self, x, max_val):
        return int(max_val * np.math.sin(2 * 3.14 * x / self.period))

def reverse_img(word_img):
    offset = np.random.randint(-10, 10)
    return 255 + offset - word_img
