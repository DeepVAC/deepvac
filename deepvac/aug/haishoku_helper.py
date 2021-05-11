from PIL import Image

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