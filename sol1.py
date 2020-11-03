import os
import sys

import numpy as np
from matplotlib import pyplot as plt
import imageio as imio
import scipy as sp
from skimage.color import rgb2gray

WRONG_PATH_MSG = "Error: bad filename"
MAX_VAL = 255
SHADE_LEVELS = 256
FIRST_IDX = 0
LAST_IDX = -1
SQUARE_POWER = 2
GRAY = 1
CHAN_1 = 0
CHAN_2 = 1
CHAN_3 = 2
YIQ = np.array([[0.299, 0.587, 0.114],
                [0.596, -0.275, -0.321],
                [0.212, -0.523, 0.311]], dtype="float64")


def read_image(filename, representation):
    """
    Reads an image into a float64 array
    :param filename: The path to the image
    :param representation: 1 if output should be grayscale, 2 if output should be RGB
    :return: The image data as ndarray
    """
    if not os.path.isfile(filename):
        sys.exit(WRONG_PATH_MSG)
    im_arr = imio.imread(filename)
    im_arr = im_arr.astype("float64")
    im_arr /= MAX_VAL
    if representation == GRAY:  # todo - check whether the pic is already grey?
        im_arr = rgb2gray(im_arr)
    return im_arr


def im_data_disp(im_arr, representation):
    """
    Displays an image
    :param im_arr: An array containing the image data
    :param representation: 1 if output should be grayscale, 2 if output should be RGB
    :return: None
    """
    if representation == GRAY:
        plt.imshow(im_arr[..., [FIRST_IDX]], cmap=plt.cm.gray)
    else:
        plt.imshow(im_arr)
    plt.show()


def imdisplay(filename, representation):
    """
    Displays an image from a given path
    :param filename: The path to the image
    :param representation: 1 if output should be grayscale, 2 if output should be RGB
    :return: None
    """
    im_data_disp(read_image(filename, representation), representation)


def rgb2yiq(imRGB):
    """
    Converts a RGB color image to YIQ grayscale image
    :param imRGB: An array containing the original image data
    :return: An array containing the modified image data
    """
    y = np.inner(YIQ[CHAN_1], imRGB)
    i = np.inner(YIQ[CHAN_2], imRGB)
    q = np.inner(YIQ[CHAN_3], imRGB)
    return np.dstack((y, i, q))


def yiq2rgb(imYIQ):
    """
    Converts a YIQ grayscale image to RGB color image
    :param imRGB: An array containing the original image data
    :return: An array containing the modified image data
    """
    inv_yiq = np.linalg.inv(YIQ)
    r = np.inner(imYIQ, inv_yiq[CHAN_1])
    g = np.inner(imYIQ, inv_yiq[CHAN_2])
    b = np.inner(imYIQ, inv_yiq[CHAN_3])
    return np.dstack((r, g, b))


def histogram_equalize(im_orig):
    """
    Preforms histogram equalization on an image
    :param im_orig: The original image
    :return: A list containing the modified image, the original histogram and the histogram after the equalization
    """
    grayscale = rgb2yiq(im_orig)  # todo - check whether the image is YIQ?
    y = (grayscale[..., CHAN_1].copy() * MAX_VAL).astype("uint8")
    orig_hist, bins = np.histogram(y, SHADE_LEVELS)
    cum_hist = np.cumsum(orig_hist)
    min_val_amount = cum_hist[orig_hist.nonzero()[FIRST_IDX][FIRST_IDX]]
    lut = (MAX_VAL * (cum_hist - min_val_amount) / (cum_hist[LAST_IDX] - min_val_amount)).round().astype("uint8")
    grayscale[..., CHAN_1] = lut[y] / lut[y].max()  # todo - is this the right division?
    res = yiq2rgb(grayscale)
    return [res, orig_hist, np.histogram(y, SHADE_LEVELS)[FIRST_IDX]]


def quantize(im_orig, n_quant, n_iter):
    """
    Preforms image quantization on an image
    :param im_orig: The original image
    :param n_quant: The num of shades to output
    :param n_iter: The num of iterations to run the optimization stage
    :return: A list containing the modified image and an array of the error of each iteration
    """
    grayscale = rgb2yiq(im_orig)
    y = (grayscale[..., CHAN_1].copy() * MAX_VAL).astype("uint8")
    orig_hist, bins = np.histogram(y, SHADE_LEVELS)
    z = np.quantile(y, np.linspace(FIRST_IDX, 1, n_quant + 1)).astype("uint8")  # todo - smart choose
    q = np.empty(n_quant)
    error = np.zeros(n_iter)
    for it in range(n_iter):
        prev_z = z.copy()
        for i in range(n_quant):
            denum = orig_hist[z[i]:z[i + 1]].sum()
            num = sum(orig_hist[cur] * cur for cur in range(z[i], z[i + 1]))
            q[i] = round(num / denum)  # todo - no round?
            error[it] += sum(orig_hist[cur] * (cur - q[i]) ** SQUARE_POWER for cur in range(z[i], z[i + 1]))
        for i in range(n_quant - 1):
            z[i] = ((q[i] + q[i + 1]) / 2)
        if (z == prev_z).all:
            break
    lut = [q[i] for i in range(n_quant) for j in range(z[i], z[i+1])]
    rem = [q[LAST_IDX]] * (SHADE_LEVELS - len(lut))
    lut = np.array(lut + rem)
    grayscale[..., CHAN_1] = lut[y] / MAX_VAL
    im_quant = yiq2rgb(grayscale)
    return [im_quant, np.trim_zeros(error)]
