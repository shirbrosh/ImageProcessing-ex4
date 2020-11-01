from scipy.signal import convolve2d
from imageio import imread
import numpy as np
import scipy.signal as sig
from scipy.ndimage.filters import convolve
import skimage.color as skimage
import math


GRAY_SCALE = 1
NORMALIZED = 255
MIN_SIZE = (16 * 16)
REDUCE_CONVOLVE = np.array([1.0, 1.0])
NORM_REDUCE_BLUR = 1 / 4


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img

def read_image(filename, representation):
    """
    This function reads an image file and converts it into a given representation.
    :param filename: the filename of an image on disk(could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the
        output should be a grayscale image(1) or an RGB image(2).
    :return: An image, represented by a matrix of type np.float64 with intensities.
    """
    image = imread(filename)

    # checks if the image is already from type float64
    if not isinstance(image, np.float64):
        image.astype(np.float64)
        image = image / NORMALIZED

    # checks if the output image should be grayscale
    if representation == GRAY_SCALE:
        image = skimage.rgb2gray(image)
    return image

def calculate_filter_vec(filter_size, norm):
    """
    This function calculates the filter vector- used for the pyramid construction
    :param filter_size: the size of the gaussian filter to be used in constructing
        the pyramid filter
    :param norm: the blur normalization
    :return: the filter vector
    """
    if filter_size == 1:
        return [[1]]
    filter_vec = REDUCE_CONVOLVE
    for i in range(filter_size - 2):
        filter_vec = np.array(sig.convolve(filter_vec, REDUCE_CONVOLVE))
    power = (filter_size - 1) / 2
    filter_vec *= math.pow(norm, power)
    return filter_vec

def reduce(im, filter_vec):
    """
    This function operates the reduce algorithm
    :param im: the image to operate the reduce on
    :param filter_vec: the vector to convolve with the image in order to blur
    :return: the image after reduce
    """
    # blur image:
    layer_blur = blur(im, filter_vec)
    # sub sample the image:
    layer_blur_sample = layer_blur[::2, 1::2]
    return layer_blur_sample


def blur(im, filter_vec):
    """
    This function operates a blur on a given image
    :param filter_vec: the vector to convolve with the image in order to blur
    :param im: the image to operate the blur on
    :return: a blurred image
    """
    layer = convolve(im, filter_vec)
    layer_blur = convolve(layer, filter_vec.T)
    return layer_blur

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function construct a gaussian pyramid on a given image
    :param im: a greyscale image with double values in [0,1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the gaussian filter to be used in constructing
        the pyramid filter
    :return: pyr - the result pyramid
            filter_vec - row vector used for the pyramid construction
    """
    filter_vec = np.reshape(calculate_filter_vec(filter_size, NORM_REDUCE_BLUR),
                            (1, filter_size))

    pyr = [im]
    blur_im = im
    i = 1
    while blur_im.size > MIN_SIZE and i < max_levels:
        layer_blur_sample = reduce(blur_im, filter_vec)
        pyr.append(layer_blur_sample)
        blur_im = layer_blur_sample
        i += 1
    return pyr, filter_vec