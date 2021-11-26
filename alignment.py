# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:44:01 2021

@author: Liz_J
"""
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy import wcs
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi


from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
import copy as cp

from skimage.registration import phase_cross_correlation, optical_flow_tvl1
from skimage import exposure
from skimage import transform
from skimage.transform import warp_polar, rotate
from skimage.filters import window, difference_of_gaussians

from scipy.fftpack import fft2, fftshift
from matplotlib.colors import LogNorm
import colorcet as cc
from astropy.convolution import convolve, Gaussian2DKernel

# from skimage.exposure import rescale_intensity
from skimage.measure import ransac

import matplotlib.gridspec as gridspec

from . import alignment_utilities as au

pcc_params = {
    'histogram_equalisation' : True,
    'remove_boundary_pixels' : 25,
    'hpf' : difference_of_gaussians,
    'hpf_kwargs': {
        'low_sigma': 0.9,
        'high_sigma':0.9*8
    },
}

# iterative_rotation_params = {
#     'max_angle' : 60, #degrees
#     'resolution' : 0.5, #degrees
#     'radius_fraction': 4, #use inner quarter of image
#     'poly_order': 7,
#     'exlucde_angles':5 #degrees

# }


class Verbose_prints:
    """
    General print statments to inform the user what the alignment algorithms
    have found. Only prints if the user asks for verbose mode to be switched
    on
    """
    def __init__(self, verbose):
        """
        Initalised the print statments

        Parameters
        ----------
        verbose : bool
            Tells the class whether to print or not.

        Returns
        -------
        None.

        """
        self.verbose = verbose
        if self.verbose:
            print('Verbose mode on')
        else:
            print('Verbose mode off')


    def default_filter_params(self):
        """
        Prints the default filter parameters if no parameters
        are set by the user to inform them how the images are filtered
        before the analysis is run on them

        Returns
        -------
        None.

        """
        if self.verbose:
            print('\nRemoving boundary of %d. Filtering image using '\
                  'histogram equalisation and '\
                  'high pass filter %s with arguments:' \
                   % (pcc_params['remove_boundary_pixels'], pcc_params['hpf'].__name__))
            for key in pcc_params['hpf_kwargs'].keys():
                print(key, pcc_params['hpf_kwargs'][key])

    def applied_translation(self, yx_offset):
        """
        Prints whenever the user manually applies a translational offset
        to the prealign image

        Parameters
        ----------
        yx_offset : 2-array
            The y and x offset applied to the prealign image.
            Positive values shift the prealign in the negative direction i.e
            A coordinate of 3,3, with an offser of 1,1 will be shifted to 2,2.

        Returns
        -------
        None.

        """
        if self.verbose  and np.sum(yx_offset) != 0:
            print('\nAdditonal offset of:\ny=%.4f, x=%4f pixels has been applied.'\
                  ' All manually given offsets recorded in'\
                  ' `self.manually_applied_offsets`' % tuple(yx_offset))

    def applied_rotation(self, rotation_angle):
        """
        Prints whenever the user manually applies a translational offset
        to the prealign image

        Parameters
        ----------
        rotation_angle : float
            Angle, in degrees, of the rotational offset applied to the prealign
            image

        Returns
        -------
        None.

        """

        if self.verbose and rotation_angle !=0:
            print('\nAdditonal rotation of:theta=%4f degrees has been applied.'\
                  ' All manually given offsets recorded in'\
                  ' `self.manually_applied_offsets`' % rotation_angle)

    def get_translation(self, shifts, added_shifts=[0,0]):
        """
        Prints the translation found by the align object

        Parameters
        ----------
        shifts : 2-array
            The y and x offset needed to align the prealign image with the
            reference image.  Positive values shift the prealign in the negative direction i.e
            A coordinate of 3,3, with an offser of 1,1 will be shifted to 2,2.
        added_shifts : 2-array, optional
            Additional shifts already applied to the image. The default is [0,0].

        Returns
        -------
        None.

        """
        if self.verbose:
            print('\n--------Found translational offset---------')
            print('Offset of %g y, %g x pixels found' % tuple(shifts))

            if np.sum(added_shifts) != 0:
                shifty = shifts[0] + added_shifts[0]
                shiftx = shifts[1] + added_shifts[1]
                print('Total offset of %g y, %g x pixels found' % (shifty, shiftx))

    def get_rotation(self, rotation, added_rotation=0):
        """
        Prints the rotational offset found by the align object

        Parameters
        ----------
        rotation : float
            Angle, in degrees,  needed to align the prealign image with the
            reference image
        added_rotation : float, optional
            Additional rotation already applied to the image. The default is 0.

        Returns
        -------
        None.

        """
        if self.verbose:
            print('\n----------Found rotational offset----------')
            print('Rotation of %.4f degrees found' % rotation)
            if added_rotation != 0:
                tot_rot = rotation + added_rotation
                print('Total offset of %.4f degrees found' % tot_rot)



class Alignment_base:
    """
    Base alignment functions most alignment methods require, such as
    filtering images (such as a high pass filter) to improve the extraction of
    features needed to estimate how features have shifted from the reference
    image to the prealign image. `guess_translation` and `guess_rotation`
    allow you to start with a guess or apply a known solution. Rotation
    is *ALWAYS* applied first, then translation. This both matches MUSE's
    implimentation and also how rotations and translations are recovered
    in image registration methods.
    """

    def __init__(self, prealign, reference, convolve_prealign=None, convolve_reference=None, guess_translation=[0,0], guess_rotation=0, verbose=True, header=None, filter_params={}):
        """
        Runs the base alignment prep comment to all child aligment classes.
        It will perform the inital filters needed to help with the aligment
        algorithms and will also apply and user given offsets.
        To keep track of any user given offsets, the applied offsets and
        the order they were applied are saved in `self.manually_applied_offsets`.
        When multiple image translations and rotations are stacked, the
        shifts and rotations do not nesacary add linearly. To ensure the absolute
        correct total alignemnt parameters are correctely stack for
        the user to align the image with a rotation, then a translation,
        use the homography matrix with attribute name `self.matrix_transform`


        Parameters
        ----------
        prealign : 2d array
            Image to be aligned.
        reference : 2d array
            Reference image with the correct alignment.
        convolve_prealign : int or None, optional
            If a number, it will convolve the prealign image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            (I think). The default is None.
        convolve_reference : int or None, optional
            If a number, it will convolve the reference image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            (I think). The default is None.
        guess_translation : 2-array, optional
            A starting translation you want to apply before running
            the alignment. The default is [0,0]. Positive values translate
            the image in the oposite direction. A value of [-3,-2] will translate
            the image upwards by three pixels and to the right by 2 pixels.
        guess_rotation : float, optional
            A starting rotation you want to apply before running. Units are in
            degrees, and a positive value rotates counter-clockwise.
            the alignment. The default is [0,0].
        verbose : bool, optional
            Tells the class whether to print out information for the user.
            The default is True.
        header : header: astropy.io.Header.header
            Dictionary-like object containing the world coordinate reference.
            The default is None.
        filter_params : dict, optional
            A dictionary containing user defined paramters for the image filtering
            If the default is {}, uses the filter paramters stored in `pcc_params`

        Returns
        -------
        None.

        """

        self.verbose = verbose
        self.print = Verbose_prints(self.verbose)
        self.header = header


        self.prealign = prealign
        self.reference = reference

        self.convolve_prealign  = convolve_prealign
        self.convolve_reference = convolve_reference

        if not filter_params:
            self.filter_params = pcc_params
            self.print.default_filter_params()
        else:
            self.filter_params = filter_params

        self.prealign_filter = self.filter_image_for_anaylsis(
            image=self.prealign,
            convolve=self.convolve_prealign,
            **self.filter_params
        )

        self.reference_filter = self.filter_image_for_anaylsis(
            image=self.reference,
            convolve=self.convolve_reference,
            **self.filter_params
        )

        self.manually_applied_offsets = {
            'rotations':[],
            'translations': []
        }

        # This is the euclid homography matrix. It is another way
        # of storing the translation and rotation. Useful since if
        # you apply a rotation and then a translation, it will accuratly track
        # the total rotation, then translation that has been applied
        #
        self.matrix_transform = np. linalg.inv(
            au.create_euclid_homography_matrix(
                guess_rotation,
                -np.array(guess_translation)
            )
        )

        #applying a guess/known translation and or rotation
        try:
            self.translate_prealign(guess_translation)

        except ValueError:
            #ValueError: Big-endian buffer not supported on little-endian compiler
            self.prealign = self.prealign.byteswap().newbyteorder()
            self.prealign_filter = self.prealign_filter.byteswap().newbyteorder()
            self.translate_prealign(guess_translation)

        self.rotate_prealign(guess_rotation)

    @classmethod
    def from_fits(cls, filename_prealign, filename_reference, hdu_index_prealign=0, hdu_index_reference=0, convolve_prealign=None, convolve_reference=None, guess_translation=[0,0], guess_rotation=0, verbose=True, filter_params={}):
        """
        Initalises the class straight from the filepaths

        Parameters
        ----------
        cls : object
            object equivlent to self.
        filename_prealign : str
            Filepath to the prealign fits file image.
        filename_reference : str
            Filepath to the reference fits file image.
        hdu_index_prealign : int or str, optional
            Index or dict name for prealign image if the hdu object has
            multiple pbjects. The default is 0.
        hdu_index_reference : int or str, optional
            Index or dict name for reference image if the hdu object has
            multiple pbjects. The default is 0.
        convolve_prealign : int or None, optional
            If a number, it will convolve the prealign image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            (I think). The default is None.
        convolve_reference : int or None, optional
            If a number, it will convolve the reference image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            (I think). The default is None.
        guess_translation : 2-array, optional
            A starting translation you want to apply before running
            the alignment. The default is [0,0]. Positive values translate
            the image in the oposite direction. A value of [-3,-2] will translate
            the image upwards by three pixels and to the right by 2 pixels.
        guess_rotation : float, optional
            A starting rotation you want to apply before running. Units are in
            degrees, and a positive value rotates counter-clockwise.
            the alignment. The default is [0,0].
        verbose : bool, optional
            Tells the class whether to print out information for the user.
            The default is True.
        filter_params : dict, optional
            A dictionary containing user defined paramters for the image filtering
            If the default is {}, uses the filter paramters stored in `pcc_params`

        Returns
        -------
        object
            Initalises the class.

        """

        data_prealign, header = au.open_fits(filename_prealign, hdu_index_prealign)
        data_reference = au.open_fits(filename_reference, hdu_index_reference)[0]

        return cls(data_prealign, data_reference, convolve_prealign, convolve_reference, guess_translation, guess_rotation, verbose, header, filter_params)

    def translate_prealign(self, yx_offset):
        """
        Transates the filtered prealign image and updates the homography matrix,
        `self.matrix_transform` needed to align the prealign to the reference

        Parameters
        ----------
        yx_offset : 2-array
            The yx coordinate to translate the image.

        Returns
        -------
        None.

        """

        self.prealign_filter = au.translate_image(self.prealign_filter, yx_offset)
        self.manually_applied_offsets['translations'].append(yx_offset)
        self.manually_applied_offsets['rotations'].append(0)

        #inverse matrix as this is the matrix needed to transform
        #the prealign image to the reference image co-ordinates
        matrix_translation = np. linalg.inv(
            au.create_euclid_homography_matrix(0, -np.array(yx_offset)
            )
        )
        self.matrix_transform = matrix_translation @ self.matrix_transform

        self.print.applied_translation(yx_offset)

    def rotate_prealign(self, rotation_angle):
        """
        Rotates the filtered prealign image and updates the homography matrix,
        `self.matrix_transform` needed to align the prealign to the reference

        Parameters
        ----------
        rotation_angle : float
            The angle, in degrees, to rotate the prealign image by.

        Returns
        -------
        None.

        """

        self.prealign_filter = au.rotate_image(self.prealign_filter, rotation_angle)
        self.manually_applied_offsets['rotations'].append(rotation_angle)
        self.manually_applied_offsets['translations'].append([0,0])

        #inverse matrix as this is the matrix needed to transform
        #the prealign image to the reference image co-ordinates
        matrix_rotation = np.linalg.inv(
            au.create_euclid_homography_matrix(rotation_angle, [0,0]
            )
        )
        self.matrix_transform = matrix_rotation @ self.matrix_transform

        self.print.applied_rotation(rotation_angle)


    def filter_image_for_anaylsis(self, image, histogram_equalisation=False, remove_boundary_pixels=25, convolve=None, hpf=None, hpf_kwargs={}):
        """
        The function that contols how the prealign and reference image are filtered
        before running the alignment

        Parameters
        ----------
        image : 2d array
            Image to apply the filters to.
        histogram_equalisation : Bool, optional
            If true, scales the intensity values according to a histogram
            equalisation. This tends to help computer vison find features
            as it maximises the contast. The default is False.
        remove_boundary_pixels : int, optional
            Removes pixels around the image if there are bad pixels at the
            detector edge. The default is 25.
        convolve : int or None, optional
            If a number, it will convolve the image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            (I think). The default is None
        hpf : function, optional
            The high pass filter function to use to filter out high frequencies.
            Higher frequencies can reduce the performance of alignment
            routines. The default is None.
        hpf_kwargs : dict, optional
            The dictionary arguments needed for `hpf`. The default is {}.

        Returns
        -------
        image : 2d array
            The filtered image.

        """


        image = self._remove_image_border(image, remove_boundary_pixels)
        image = self._remove_nonvalid_numbers(image)
        if convolve is not None:
            image = au.convolve_image(image, convolve)

        #this helps to match up the intensities
        if histogram_equalisation:
            image = exposure.equalize_hist(image)

        # remove some frequencies to make alignment more robust vs noise
        if hpf is not None:
            image = hpf(image, **hpf_kwargs)

        return image


    def _remove_image_border(self, image, border):
        """
        Shrinks the image by removing the given border value long each edge
        of the image

        Parameters
        ----------
        image : 2darray
            Image that will have its borders removed
        border : int
            Number of pixels to remove from each edge.


        Returns
        -------
        image_border_removed : 2darray
            Image with boundary pixels removed Returned in a
            nd-2*border, nd-2*border array.

        """
        # data_shape = min(np.shape(image))
        # image = cp.deepcopy(image[:data_shape, :data_shape])

        image_border_removed = image[border:-border, border:-border]

        return image_border_removed

    def _remove_nonvalid_numbers(self, image):
        """
        Removes non valid numbers from the array and replaces them with zeros

        Parameters
        ----------
        image : 2darray
            Image that will have its nonvalid values replaced with zeros

        Returns
        -------
        image_valid : 2darray
            Image without any NaNs or infs

        """
        # nonvalid_mask = ~np.isfinite(image)
        # image[nonvalid_mask] = 0
        image = cp.deepcopy(np.nan_to_num(image))

        return image


class Align_homography():
    """
    Finds the offse between two images by estimating the matrix transform needed
    to reproduce the vector changes between a set of xy coordinates.

    This can align an image given a minimum of 4 xy, grid points from the reference
    image and their location in the prealign image.

    Default homography is the skimage.transform.EuclideanTransform. This forces
    the solution to output only a rotation and a translation.
    Other homography matrices can be used to find image shear, and scale changed
    between th reerence and prealign image.

    Homography works because the translation and rotation (and scale etc) matix
    can be combined into one operation. This matrix is called the homography
    matrix. We can then write out a set of equaltions
    describing the new yx, grid points as a function of the rotation and translation.
    By writing these equations for the new x and y for each grid point
    up as a matrix, we can just use singular value decomosition (SVD) to find the
    coefficents (ie the translation and rotation) and out.
    To improve this, we typically use least squares, with additional constraints
    that we now to help improve the outputed homography matrix.

    Most off the shell homography routines apply the rotation matrix first
    then apply the translation matrix. To reverse this ordering, set
    `self.reverse_homography` in the method `homography_on_grid_points`
    to True

    """

    def correct_homography_order(self, homograpghy_matrix):
        """
        Homography order is rotation then translation. Since rotation and then
        translation are non-commutive, the homography matrix would need
        to change if translation, then rotation is applied. If translation
        then rotation was applied, use this method to adjust off-the-shelf
        homopgraphy finding method to match this order of transformation.
        This only works for Euclidean tranforms (i.e those limited to
        rotation and translation) If affine/projective etc tranforms have been
        used, this will not be the correct solution

        Parameters
        ----------
        homograpghy_matrix : 3x3 matrix
            Euclidean transform homogrpahy matrix containing the rotational
            and translational information.

        Returns
        -------
        2-array
            Returns the adjusted shifts for translation followed by rotation.

        """

        a = homograpghy_matrix[0,-1]
        b = homograpghy_matrix[1,-1]
        cos_theta = homograpghy_matrix[0, 0]
        sin_theta = homograpghy_matrix[1, 0]

        dy = b * cos_theta - a * sin_theta
        dx = a * cos_theta + b * sin_theta

        return np.array([dy, dx])

    def homography_on_grid_points(self, origonal_xy, tranformed_xy,
                                  method=transform.EuclideanTransform,
                                  reverse_homography=False,
                                  ):
        """
        Performs a fit to estimate the homography matix that represents the grid
        transformation. Uses ransac to robustly elimiate vector outliers that
        something such as optical flow, or other feature extracting methods
        might output. The parameters are not currently chanable by the user
        since not entirely sure what are the best paramters to use and so
        are using values given in skimage turorials.
        TODO: Make method work with other transforms. Currently only works
        for Euclidean transformations (i.e., pure rotations and translations)

        Parameters
        ----------
        origonal_xy : nx2 array
            nx2 array of the xy coordinates of the reference grid.
        tranformed_xy :  nx2 array
            nx2 array of the xy coordinates of a feature that has been transformed
            by a rotation and translation (coordinates of object in prealign grid)
        method : function, optional
            method is the what function is used to find the homography.
            The default is transform.EuclideanTransform.
        reverse_homography : bool, optional
            Homograpgy matrix outputs the solution representing a rotation followed
            by a translation. If you want the solution to translate, then rotate
            set this to True. The default is False.

        Returns
        -------
        self.shifts : 2-array
            The recovered shifts, in pixels.
        self.rotation : float
            The recovered rotation, in degrees.
        self.homography_matrix : 3x3 matrix
            The Matrix that has been "performed" that resulted in the offset
            between the prealign and the reference. The inverse therefore
            describes the paramters needed to convert the prealign image
            to the reference grid.

        """

        model = method()
        found = model.estimate(origonal_xy, tranformed_xy)
        #robust outlier removal. Since I don't know what changing these
        #paramters does, I've kept them as fixed and unaccessable to
        #user for now. The output solution changes per run using ransac
        model_robust, inliers = ransac((origonal_xy, tranformed_xy), method, min_samples=3, residual_threshold=2, max_trials=100)

        #If you want the matrix to represent translation then rotation, set this
        #to True
        if reverse_homography:
            shifts = self.correct_homography_order(model_robust.params)
        else:
            shifts = np.array([model_robust.params[1,-1], model_robust.params[0,-1]])

        self.homography_matrix = model_robust.params
        self.rotation = np.arcsin( model_robust.rotation) * 180 / np.pi
        self.shifts = -1 * shifts

        return self.shifts, self.rotation, self.homography_matrix

class Align_optical_flow(Alignment_base, Align_homography):
    """
    Optical flow is a gradient based method that find small changes between
    two images as vectors for each pixel. Recomended for translations and
    rotations that are less than 5 pixels. If larger than this, use
    the iterative method.

    To do optical flow, the intensties between the two image much be as equal
    as posible. Changes in intensity are kinda interpeted as a scale change.
    To match the intensities, histogram equalisation, or some intensity
    scaling is needed.
    """

    def optical_flow(self):
        """
        Performs the optical flow.

        Returns
        -------
        v : 2d array
            y component of the vectors describing the offset between the prealign
            and the reference image.
        u : 2d array
            x component of the vectors describing the offset between the prealign
            and the reference image.

        """

        v, u = optical_flow_tvl1(self.reference_filter, self.prealign_filter)

        return v, u

    def _prep_arrays(self, num_per_dimension):
        """
        Gets a sample of coordinates from the optical flow to perform
        homography on. A sample is needed because the full image is expensive
        to run.

        Parameters
        ----------
         num_per_dimension : int
             For each axis, this sets the number of grid points to be calcualted.

        Returns
        -------
        xy_stacked : num_per_dimensionx2 array
            The paired xy coordinates of the vector origin.
        xy_stacked_pre : num_per_dimensionx2 array
            The paired xy coordinates of the vector end.

        """
        #Vector field. V are the y component of the vectors, u are the x components
        self.v, self.u = self.optical_flow()

        y_sparse, x_sparse, v_sparse, u_sparse = au.get_sparse_vector_grid(self.v, self.u, num_per_dimension)

        y_sparse_pre = y_sparse + v_sparse
        x_sparse_pre = x_sparse + u_sparse

        xy_stacked = au.column_stack_2d(x_sparse, y_sparse)
        xy_stacked_pre = au.column_stack_2d(x_sparse_pre, y_sparse_pre)

        return xy_stacked, xy_stacked_pre

    def get_translation_rotation_single(self, num_per_dimension=50, homography_method=transform.EuclideanTransform, reverse_homography=False):
        """
        Works out the translation and rotation using homography once

        Parameters
        ----------
         num_per_dimension : int, optional
             For each axis, this sets the number of grid points to be calcualted.
The default is 50.
        homography_method : function, optional
            The (skimage) transformation method. Different transforms can find
            additional types of matrix transformations that might be present.
            For alignment, we only want to find rotation and transformation. For
            only these transfomations, use the Euclidean transform. The default
            is transform.EuclideanTransform.
        reverse_homography : bool, optional
            If the order that a user will use to  correct the alignment is NOT
            rotation, then translation, set this to True. The default is False.

        Returns
        -------
        self.shifts : 2-array
            The recovered shifts, in pixels.
        self.rotation : float
            The recovered rotation, in degrees.
        self.homography_matrix : 3x3 matrix
            The Matrix that has been "performed" that resulted in the offset
            between the prealign and the reference. The inverse therefore
            describes the paramters needed to convert the prealign image
            to the reference grid.

        """

        reference_grid_xy, prealign_grid_xy = self._prep_arrays(num_per_dimension)

        #Using homography
        self.shifts, self.rotation, self.homography_matrix = self.homography_on_grid_points(
            origonal_xy=reference_grid_xy,
            tranformed_xy=prealign_grid_xy,
            method=homography_method, #must be a method that outputs the 3x3 homography matrix
            reverse_homography=reverse_homography
        )
        added_rotations = np.sum(self.manually_applied_offsets['rotations'])
        added_translations = np.sum(self.manually_applied_offsets['translations'], axis=0)

        self.print.get_rotation(self.rotation, added_rotations)
        self.print.get_translation(self.shifts, added_translations)

        self.shifts = self.shifts + added_translations
        self.rotation += added_rotations

        return self.shifts, self.rotation, self.homography_matrix

    def get_translation_rotation(self, itera=20, num_per_dimension=50, homography_method=transform.EuclideanTransform, reverse_homography=False):
        """
        If the solution is offset by more than ~5 pixels, optical flow struggles
        to match up the pixels. This method gets around this by applying optical
        flow multiple times to that the shifts and rotations converge to a solution.
        TODO: Make this stop after a the change between solutions is smaller that
        a user defined value (i.e >1% for example)

        Parameters
        ----------
        itera : int, optional
            The number of times to perform optical flow to find the shifts and
            rotation. The default is 20.
         num_per_dimension : int, optional
             For each axis, this sets the number of grid points to be calcualted.
The default is 50.
        homography_method : function, optional
            The (skimage) transformation method. Different transforms can find
            additional types of matrix transformations that might be present.
            For alignment, we only want to find rotation and transformation. For
            only these transfomations, use the Euclidean transform. The default
            is transform.EuclideanTransform.
        reverse_homography : bool, optional
            If the order that a user will use to  correct the alignment is NOT
            rotation, then translation, set this to True. The default is False.


        Returns
        -------
        self.shifts : 2-array
            The recovered shifts, in pixels.
        self.rotation : float
            The recovered rotation, in degrees.

        """
        # h_inv = self.matrix_transform
        for i in range(itera):
            #This finds
            __, __, homography_matrix = self.get_translation_rotation_single(num_per_dimension, homography_method, reverse_homography)
            shifts = -1 * np.array([homography_matrix[1,-1], homography_matrix[0,-1]])
            rotations = np.arcsin(homography_matrix[1,0]) * 180 / np.pi

            h_inv = np.linalg.inv(homography_matrix)

            #updating the prealign image, this also updates `self.matrix_transform`
            self.translate_prealign(shifts)
            self.rotate_prealign(rotations)

        self.shifts = [self.matrix_transform[1,-1], self.matrix_transform[0,-1]]
        self.rotation = np.arcsin(-self.matrix_transform[1,0]) * 180 / np.pi

        return self.shifts, self.rotation

if __name__ == "__main__":
    pass
