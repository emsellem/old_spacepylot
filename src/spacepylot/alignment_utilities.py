# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:50:06 2021

@author: Liz_J
"""
__author__ = "Elizabeth Watkins"
__copyright__ = "Elizabeth Watkins"
__license__   = "MIT License"
__contact__ = "<liz@email"

from astropy.io import fits
import numpy as np
from scipy import ndimage as ndi

import copy as cp

from skimage.transform import rotate

from astropy.convolution import convolve, Gaussian2DKernel


def column_stack_2d(array_2d_first, array_2d_second):
    """
    Converts equal dimension 2d arrays into index matched values

    Parameters
    ----------
    array_2d_first : 2d array
        Th array to fill column 1.
    array_2d_second : 2d array
        Th array to fill column 2.

    Returns
    -------
    array_stacked : n*mx2 array
        Contains the column stacked values of the given arrays.

    """

    array_2d_first_flat = array_2d_first.flatten()
    array_2d_second_flat = array_2d_second.flatten()

    array_stacked = np.column_stack([array_2d_first_flat, array_2d_second_flat])

    return array_stacked


def create_euclid_homography_matrix(rotation, translation, rotation_first=True):
    """
    Creates the homography representation of a given rotation and translation.

    Parameters
    ----------
    rotation : float
        Angle, in degrees, of the rotational offset
    translation : 2-array
        An array containing 2 numbers. First number is the y shift to
        be applied, second is the x
    rotation_first : bool, optional
        Homography perform rotation first, the translation. To reverse this,
        set bool to False. The default is True.

    Returns
    -------
    homography : 3x3 array
        The 3x3 homography matrix representing the rotation and translation
        of an image

    """
    y, x = translation
    rotation_radians = np.deg2rad(rotation)
    cos_theta, sin_theta = np.cos(rotation_radians), np.sin(rotation_radians)

    if rotation_first:
        a, b = x, y
    else:
        a = x * cos_theta + y * sin_theta
        b = y * cos_theta - x * sin_theta

    homography = np.array([
        [cos_theta, -sin_theta, a],
        [sin_theta, cos_theta, b],
        [0, 0, 1]
        ]
    )
    return homography


def correct_homography_order(homography_matrix):
    """
    Homography order is rotation then translation. Since rotation and then
    translation are non-commutative, the homography matrix would need
    to change if translation, then rotation is applied. If translation
    then rotation was applied, use this method to adjust off-the-shelf
    homography finding method to match this order of transformation.
    This only works for Euclidean transforms (i.e., those limited to
    rotation and translation) If affine/projective etc. transforms have been
    used, this will not be the correct solution

    Parameters
    ----------
    homography_matrix : 3x3 matrix
        Euclidean transform homography matrix containing the rotational
        and translational information.

    Returns
    -------
    2-array
        Returns the adjusted shifts for translation followed by rotation.

    """

    a = homography_matrix[0, -1]
    b = homography_matrix[1, -1]
    cos_theta = homography_matrix[0, 0]
    sin_theta = homography_matrix[1, 0]

    dy = b * cos_theta - a * sin_theta
    dx = a * cos_theta + b * sin_theta

    return np.array([dy, dx])


def translate_image(image, shifts):
    """
    Applies simple translation shift while preserving the input dimensions

    Parameters
    ----------
    image : 2darray
        Image to apply translational offset to.
    shifts : y,x array of 2 float
        An array containing 2 numbers. First number is the y shift to
        be applied, second is the x

    Returns
    -------
    image_translated : 2darray
        Image with new coordinate grid translated according to given
        `shifts`. Input dimensions preserved

    """
    image_no_nans = np.nan_to_num(image)
    image_translated = cp.deepcopy(ndi.shift(image_no_nans, shifts))

    return image_translated


def rotate_image(image, angle_degrees):
    """
    Applies simple rotational shift while preserving the input dimensions

    Parameters
    ----------
    image : 2darray
        Image to apply rotational offset to.
    angle_degrees : float
        Angle, in degrees, of the rotational offset to be applied to `image`.

    Returns
    -------
    image_rotated : 2darray
        Image with new coordinate grid rotated according to given
        `angle_degrees`. Input dimensions preserved.

    """
    image_no_nans = np.nan_to_num(image)
    image_rotated = cp.deepcopy(rotate(image_no_nans, angle_degrees))

    return image_rotated


def convolve_image(image, sigma):
    """
    Convolve the image with the sigma supplied. Sigma I think, related
    to the pixel size of the image

    Parameters
    ----------
    image : 2d-array
        Image to be convolved.
    sigma : float
        Standard deviation of the Gaussian used to convolve the image with.
        I think the units used by the function are in number of image pixels

    Returns
    -------
    image : 2d-array
        Convolved image.

    """
    kernel = Gaussian2DKernel(x_stddev=sigma)
    image = cp.deepcopy(convolve(image, kernel))

    return image


def sparse_2d_grid(array_2d, num_per_dimension=20, return_step=False):
    """
    From a given 2d array, generates a grid of y, x indices with steps
    determined by the number of wanted grid points per dimension.

    Parameters
    ----------
    array_2d : 2darray
        2d data array to find y, x grid.
    num_per_dimension : int, optional
        For each axis, this sets the number of grid points to be calculated.
        The default is 20
    return_step : bool, optional
        The step size between grid points needed to display the given
        `num_per_dimension` amount of grid points. The default is False.

    Returns
    -------
    y_inds_sparse : 2darray
        grid points int the y direction sampling the given 2darray according
        to the number of grid points requested with parameter `num_per_dimension`
    x_inds_sparse : 2darray
        grid points int the x direction sampling the given 2darray according
        to the number of grid points requested with parameter `num_per_dimension`
    step : int, optional
        Optional parameter to be returned if `return_step` is set to True.
        Returned is the step size between grid points used when initialising
        the grid

    """
    rows, cols = array_2d.shape
    smallest = min(rows, cols)
    if num_per_dimension > smallest:
        num_per_dimension = smallest

    step = max(rows//num_per_dimension, cols//num_per_dimension)

    y_inds_sparse, x_inds_sparse = np.mgrid[:rows:step, :cols:step]
    if return_step:
        return y_inds_sparse, x_inds_sparse, step
    else:
        return y_inds_sparse, x_inds_sparse


def get_sparse_vector_grid(v, u, num_per_dimension=20):
    """
    Gets a sparse grid for vector components along with the sparse yx
    positions of the grid. This is needed when calculating homography from
    the vector grid to speed up the operation and needed when
    visualising/displaying the vector field

    Parameters
    ----------
    v : 2darray
        Vector component in the y direction.
    u : 2darray
        Vector component in the x direction.
    num_per_dimension : int, optional
        For each axis, this sets the number of grid points to be calculated.
        The default is 20

    Returns
    -------
    y_sparse : 2darray
        2d array of shape `num_per_dimension`, `num_per_dimension`. containing
        the y coordinates sampled from the vector grid.
    x_sparse : 2darray
        2d array of shape `num_per_dimension`, `num_per_dimension`. containing
        the x coordinates sampled from the vector grid.
    v_sparse : 2darray
        2d array of shape `num_per_dimension`, `num_per_dimension`. containing
        the y component vectors sampled from the y components of the vector grid.
    u_sparse : 2darray
        2d array of shape `num_per_dimension`, `num_per_dimension`. containing
        the x component vectors sampled from the y components of the vector grid.

    """
    y_sparse, x_sparse, step = sparse_2d_grid(
        array_2d=u,
        num_per_dimension=num_per_dimension,
        return_step=True
    )

    u_sparse = u[::step, ::step]
    v_sparse = v[::step, ::step]

    return y_sparse, x_sparse, v_sparse, u_sparse


def open_fits(filepath, hdu_i=0):
    with fits.open(filepath) as hdulist:
        data = hdulist[hdu_i].data
        header = hdulist[hdu_i].header
    return data, header


if __name__ == "__main__":
    pass
