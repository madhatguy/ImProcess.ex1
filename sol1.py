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
GRAY = 1
RED_CHAN = 0
GREEN_CHAN = 1
BLUE_CHAN = 2
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
    im_arr /= 3  # todo - delete?
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
        plt.imshow(im_arr, cmap=plt.cm.gray)  # todo - change to use only Y channel
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
    y = np.inner(YIQ[RED_CHAN], imRGB)
    i = np.inner(YIQ[GREEN_CHAN], imRGB)
    q = np.inner(YIQ[BLUE_CHAN], imRGB)
    return np.dstack((y, i, q))


def yiq2rgb(imYIQ):
    """
    Converts a YIQ grayscale image to RGB color image
    :param imRGB: An array containing the original image data
    :return: An array containing the modified image data
    """
    inv_yiq = np.linalg.inv(YIQ)
    r = np.inner(imYIQ, inv_yiq[RED_CHAN])
    g = np.inner(imYIQ, inv_yiq[GREEN_CHAN])
    b = np.inner(imYIQ, inv_yiq[BLUE_CHAN])
    return np.dstack((r, g, b))


def histogram_equalize(im_orig):
    """
    Preforms histogram equalization on an image
    :param im_orig: The original image
    :return: A list containing the modified image, the original histogram and the histogram after the equalization
    """
    grayscale = rgb2yiq(im_orig)  # todo - check whether the image is YIQ?
    y = (grayscale[..., RED_CHAN].copy() * MAX_VAL).astype("uint8")
    orig_hist, bins = np.histogram(y, SHADE_LEVELS)
    cum_hist = np.cumsum(orig_hist)
    min_val_amount = cum_hist[orig_hist.nonzero()[0][0]]
    lut = (MAX_VAL * (cum_hist - min_val_amount) / (cum_hist[-1] - min_val_amount)).round().astype("uint8")
    grayscale[..., RED_CHAN] = lut[y] / lut[y].max()  # todo - is this the right division?
    res = yiq2rgb(grayscale)
    return [res, orig_hist, np.histogram(y, SHADE_LEVELS)[0]]


def quant_error(orig_hist, z, y, q, i):
    """

    :param orig_hist:
    :param z:
    :param y:
    :param q:
    :return:
    """


def quantize(im_orig, n_quant, n_iter):
    """
    Preforms image quantization on an image
    :param im_orig: The original image
    :param n_quant: The num of shades to output
    :param n_iter: The num of iterations to run the optimization stage
    :return: A list containing the modified image and an array of the error of each iteration
    """
    grayscale = rgb2yiq(im_orig)
    y = (grayscale[..., RED_CHAN].copy() * MAX_VAL).astype("uint8")
    orig_hist, bins = np.histogram(y, SHADE_LEVELS)
    z = np.quantile(y, np.linspace(0, 1, n_quant + 1)).astype("uint8")  # todo - smart choose
    q = np.empty(n_quant)
    print(z)
    error = np.zeros(n_iter)
    for it in range(n_iter):
        prev_z = z.copy()
        for i in range(n_quant):
            denum = orig_hist[z[i]:z[i + 1]].sum()
            num = sum(orig_hist[cur] * cur for cur in range(z[i], z[i + 1]))
            q[i] = round(num / denum)  # todo - no round?
            error[it] += sum(orig_hist[cur] * (cur - q[i]) ** 2 for cur in range(z[i], z[i + 1]))
            if np.isnan(q[i]):  # todo - why nan when n_quant == 20?
                print(z[i])
                print(orig_hist[z[i]:z[i + 1]])
                return [None, None]
        print(z)
        for i in range(n_quant - 1):
            z[i] = ((q[i] + q[i + 1]) / 2)
        if (z == prev_z).all:
            break
    lut = [q[i] for i in range(n_quant) for j in range(z[i], z[i+1])]
    rem = [q[-1]] * (SHADE_LEVELS - len(lut))
    lut = np.array(lut + rem)
    print(lut.shape)
    grayscale[..., RED_CHAN] = lut[y] / MAX_VAL
    im_quant = yiq2rgb(grayscale)
    return [im_quant, np.trim_zeros(error)]
