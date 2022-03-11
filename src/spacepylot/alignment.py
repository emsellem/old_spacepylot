# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:44:01 2021
Updated 2021-2022 - EE

@author: Liz_J
"""
__author__ = "Elizabeth Watkins"
__copyright__ = "Elizabeth Watkins"
__license__ = "MIT License"
__contact__ = "<liz@email>"


# Importing modules
import numpy as np

# Skimage
from skimage.registration import phase_cross_correlation, optical_flow_tvl1
from skimage import transform
from skimage.measure import ransac

# Internal calls
from . import alignment_utilities as au
from .alignment_utilities import (correct_homography_order, 
                                  get_shifts_from_homography_matrix,
                                  get_rotation_from_homography_matrix)
from .utils import VerbosePrints, filter_image_for_analysis
from .params import pcc_params


class AlignmentBase(object):
    """
    Base alignment functions most alignment methods require, such as
    filtering images (such as a high pass filter) to improve the extraction of
    features needed to estimate how features have shifted from the reference
    image to the prealign image. `guess_translation` and `guess_rotation`
    allow you to start with a guess or apply a known solution. Rotation
    is *ALWAYS* applied first, then translation. This both matches MUSE's
    implementation and also how rotations and translations are recovered
    in image registration methods.
    """

    def __init__(self, prealign, reference, convolve_prealign=None,
                 convolve_reference=None, guess_translation=[0, 0],
                 guess_rotation=0, verbose=True, header=None,
                 filter_params=None):
        """
        Runs the base alignment prep comment to all child alignment classes.
        It will perform the initial filters needed to help with the alignment
        algorithms and will also apply and user given offsets.
        To keep track of any user given offsets, the applied offsets and
        the order they were applied are saved in `self.manually_applied_offsets`.
        When multiple image translations and rotations are stacked, the
        shifts and rotations do not necessary add linearly. To ensure the absolute
        correct total alignment parameters are correctly stack for
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
            the image in the opposite direction. A value of [-3,-2] will translate
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
            A dictionary containing user defined parameters for the image filtering
            If the default is {}, uses the filter parameters stored in `pcc_params`

        Returns
        -------
        None.

        """

        self.verbose = verbose
        self.print = VerbosePrints(self.verbose)
        self.header = header

        self.prealign = prealign
        self.reference = reference

        self.convolve_prealign = convolve_prealign
        self.convolve_reference = convolve_reference

        if filter_params is None:
            self.filter_params = pcc_params
            self.print.default_filter_params()
        else:
            self.filter_params = filter_params

        self.prealign_filter = filter_image_for_analysis(image=self.prealign,
                                                         convolve=self.convolve_prealign,
                                                         **self.filter_params)

        self.reference_filter = filter_image_for_analysis(image=self.reference,
                                                          convolve=self.convolve_reference,
                                                          **self.filter_params)

        self.manually_applied_offsets = {
            'rotations': [],
            'translations': []
        }

        # This is the euclid homography matrix. It is another way
        # of storing the translation and rotation. Useful since if
        # you apply a rotation and then a translation, it will accurately track
        # the total rotation, then translation that has been applied
        #
        self.matrix_transform = np.identity(3)

        # Applying a guess/known translation and or rotation

        # First the rotation
        if guess_rotation != 0:
            try:
                self.rotate_prealign(guess_rotation)

            except ValueError:
                # ValueError: Big-endian buffer not supported on little-endian compiler
                self.prealign = self.prealign.byteswap().newbyteorder()
                self.prealign_filter = self.prealign_filter.byteswap().newbyteorder()
                self.rotate_prealign(guess_rotation)

        # Then the translation
        if sum(guess_translation) != 0:
            try:
                self.translate_prealign(guess_translation)

            except ValueError:
                # ValueError: Big-endian buffer not supported on little-endian compiler
                self.prealign = self.prealign.byteswap().newbyteorder()
                self.prealign_filter = self.prealign_filter.byteswap().newbyteorder()
                self.translate_prealign(guess_translation)

    @classmethod
    def from_fits(cls, filename_prealign, filename_reference, hdu_index_prealign=0,
                  hdu_index_reference=0, convolve_prealign=None, convolve_reference=None,
                  guess_translation=[0, 0], guess_rotation=0, verbose=True,
                  **filter_params):
        """
        Initialises the class straight from the filepaths

        Parameters
        ----------
        cls : object
            object equivalent to self.
        filename_prealign : str
            Filepath to the prealign fits file image.
        filename_reference : str
            Filepath gh repo clone emsellem/spacepylotto the reference fits file image.
        hdu_index_prealign : int or str, optional
            Index or dict name for prealign image if the hdu object has
            multiple objects. The default is 0.
        hdu_index_reference : int or str, optional
            Index or dict name for reference image if the hdu object has
            multiple objects. The default is 0.
        convolve_prealign : int or None, optional
            If a number, it will convolve the prealign image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            (I think). The default is None.
        convolve_reference : int or None, optional
            If a number, it will convolve the reference image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            The default is None.
        guess_translation : 2-array, optional
            A starting translation you want to apply before running
            the alignment. Positive values translate the image in the opposite
            direction. A value of [-3,-2] will translate the image upwards by
            the image upwards by three pixels and to the right by 2 pixels.
        guess_rotation : float, optional
            A starting rotation you want to apply before running. Units are in
            degrees, and a positive value rotates counter-clockwise.
            The default is 0.
        verbose : bool, optional
            Tells the class whether to print out information for the user.
            The default is True.
        filter_params : dict, optional
            A dictionary containing user defined parameters for the image filtering
            If the default is {}, uses the filter parameters stored in `pcc_params`

        Returns
        -------
        object
            Initialises the class.

        """

        data_prealign, header = au.open_fits(filename_prealign, hdu_index_prealign)
        data_reference = au.open_fits(filename_reference, hdu_index_reference)[0]

        return cls(data_prealign, data_reference, convolve_prealign,
                   convolve_reference, guess_translation,
                   guess_rotation, verbose, header, filter_params)

    def _update_transformation_matrix(self, new_rotation=0, new_translation=[0, 0]):
        """
        Updates the transformation matrix to the new solution found

        Parameters
        ----------
        new_rotation : int, optional
            The new rotation. Units are in degrees, and a positive value rotates
            counter-clockwise. The default is 0.
        new_translation : 2-array, optional
            The new translation.  Positive values translate the image in the
            opposite direction. A value of [-3,-2] will translate the image
            upwards by three pixels and to the right by 2 pixels.
            The default is [0,0].

        Returns
        -------
        None

        """
        # new_matrix = np. linalg.inv(
        #     au.create_euclid_homography_matrix(new_rotation, -np.array(new_translation)
        #     )
        # )
        new_matrix = au.create_euclid_homography_matrix(
            new_rotation,
            -np.array(new_translation),
            rotation_first=True)

        self.matrix_transform = new_matrix @ self.matrix_transform

    def translate_prealign(self, yx_offset):
        """
        Translates the filtered prealign image and updates the homography matrix,
        `self.matrix_transform` needed to align the prealign to the reference.

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

        # Inverse matrix as this is the matrix needed to transform
        # the prealign image to the reference image co-ordinates
        self._update_transformation_matrix(0, yx_offset)

        self.print.applied_translation(yx_offset)

    def rotate_prealign(self, rotation_angle):
        """
        Rotates the filtered prealign image and updates the homography matrix,
        `self.matrix_transform` needed to align the prealign to the reference.
        Everytime this is applied, the transformation matrix
        `self.matrix_transform` and lists that keep track of the
        exact order translations and rotations have been apllied (used for
        bug checking) -- `self.manually_applied_offsets` -- are updated.

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
        self.manually_applied_offsets['translations'].append([0, 0])

        # Inverse matrix as this is the matrix needed to transform
        # the prealign image to the reference image co-ordinates
        self._update_transformation_matrix(rotation_angle, [0, 0])

        self.print.applied_rotation(rotation_angle)


class AlignmentCrossCorrelate(AlignmentBase):
    """
    Finds the translational offset between the reference and prealign images
    using phase cross correlation. Phase cross correlation works by converting
    the images into fourier space. Here, translational offsets are represented
    as a phase difference, where there is no signal when they are out of phase,
    and a sharp peak when the image is in phase. This sharp peak is used
    to work out the offset between two images.
    """

    def fft_phase_correlation(self, prealign, reference, resolution=1000, **kwargs):
        """
        Method for performing the cross correlation between two images.

        Parameters
        ----------
        prealign : 2d array
            Image to be aligned.
        reference : 2d array
            Reference image with the correct alignment.
        resolution : int, optional
            Determines how precise the algorthm will run. For example, 10 will
            find a solution closest to the first decimal place, 100 the second
            etc. The default is 1000.
        **kwargs : skimage.registration.phase_cross_correlation properties, optional
            kwargs are optional parameters for
            skimage.registration.phase_cross_correlation

        Returns
        -------
        self.shifts : 2-array
            The offset needed to align the prealign image with the
            reference image.  Positive values shift the prealign in the negative
            direction i.e A coordinate of 3,3, with an offset of 1,1
            will be shifted to 2,2. If the image have been prepared properly
            the returned shifts encode rotational offsets and scale changes
            between the two images.

        """

        upsample_factor = kwargs.pop('upsample_factor', resolution)

        shifts, error,  phasediff = phase_cross_correlation(
            reference_image=reference,
            moving_image=prealign,
            upsample_factor=upsample_factor, **kwargs
            )
        return shifts


class AlignTranslationPCC(AlignmentCrossCorrelate):
    """
    Class used to find only the translational offset between a pair of
    images using phase cross correlation.
    """

    def get_translation(self, split_image=1, over_shifted=20, resolution=1000,
                        **kwargs):
        """
        Method to perform phase cross correlation to find translational offset
        between two images of equal size. Artifacts can affect the performance
        of this method, therefore to make this method more robust,
        the images can be split into quarters (2) 9ths (3) etc, and
        the method is run independently in each section. Any offset found
        that has a truly bad solution (as defined by the `over_shifted`
        parameter (units of pixels), are ignored. Final solution returns
        the median solution of all valid sections.

        Parameters
        ----------
        split_image : int, optional
            The number of subsections used to find the offset solution.
            1 uses the original image, 2 splits the image 2x2=4 times,
            3  is 3x3=9 times etc. The default is 1.
        over_shifted : float or int, optional
            Defines the limit on what is considered a plausible offset
            to find in either the x or y directions (in number of pixels).
            Any offset in x or y that is larger than this value are ignored.
            The default is 20.
        resolution : int, optional
            Determines how precise the algorithm will run. For example, 10 will
            find a solution closest to the first decimal place, 100 the second
            etc. The default is 1000.
        **kwargs : skimage.registration.phase_cross_correlation properties, optional
            kwargs are optional parameters for
            skimage.registration.phase_cross_correlation

        Returns
        -------
        self.shifts : 2-array
            The recovered shifts, in pixels.

        """

        if self.verbose:
            print('\nSplitting up image into %dx%d parts and returning median'
                  ' offset found for all panels.' %(split_image, split_image))
            print('A pixel shift > %.2f pixels in either the x or the y direction '
                  'will be ignored from the final offset calculation.' % over_shifted)

        self.all_shifts = []
        for i in range(split_image):
            for j in range(split_image):
                split_prealign, split_reference = self._get_split_images(split_image, (i, j))
                self.shifts = self.fft_phase_correlation(split_prealign, split_reference, resolution, **kwargs)

                # Ignores pixels translations that are too large
                if any(np.abs(self.shifts) > over_shifted):
                    continue
                else:
                    self.all_shifts.append(self.shifts)

        # Taking the median
        self.shifts = np.median(self.all_shifts, axis=0)
        # Update the matrix
        self._update_transformation_matrix(0, self.shifts)

        # Printing out the result
        added_shifts = np.sum(self.manually_applied_offsets['translations'], axis=0)
        self.print.get_translation(self.shifts, added_shifts)

        # Now adding the final shifts
        self.shifts += added_shifts

        return self.shifts

    def _split(self, size, num):
        """
        For a given length, and division position within an axis, works
        out the start and end indices of that sub-section of the axis.
        If an image has been subdivided 2 times (4 quadrants), so
        the length of the quadrant sides are axis/2, `num`=0 is used
        to get the start and end indices of that quadrant.

        Parameters
        ----------
        size : int
            Length of subaxis.
        num : int
            start location of subaxis.

        Returns
        -------
        lower : int
            Lower index position of a given length.
        upper : int
            Upper index position of a given length.

        """
        lower, upper = int(size) * np.array([num, num+1])
        return lower, upper

    def _get_split_images(self, split_image, nums):
        """
        Returns subsection of a given image split into smaller sections

        Parameters
        ----------
        split_image : int
            The number of subsections used to find the offset solution.
            1 uses the original image, 2 splits the image 2x2=4 times,
            3  is 3x3=9 times etc. The default is 1.
        nums : 2-array of ints
            the ith and jth subsection location.

        Returns
        -------
        split_pre : 2darray
            the ith, jth subsection of the prealigned image
        split_ref : 2darray
            the corresponding ith, jth subsection of the reference image.

        """

        split_size = np.array(np.shape(self.reference_filter)) / split_image

        [lower_y, upper_y], [lower_x, upper_x] = [self._split(split_size[n], nums[n])
                                                  for n in [0, 1]]

        split_pre = self.prealign_filter[lower_y:upper_y, lower_x:upper_x]
        split_ref = self.reference_filter[lower_y:upper_y, lower_x:upper_x]

        return split_pre, split_ref


class AlignHomography(object):
    """Finds the offset between two images by estimating the matrix transform needed
    to reproduce the vector changes between a set of xy coordinates.

    This can align an image given a minimum of 4 xy, grid points from the reference
    image and their location in the prealign image.

    Default homography is the skimage.transform.EuclideanTransform. This forces
    the solution to output only a rotation and a translation.
    Other homography matrices can be used to find image shear, and scale changed
    between th reference and prealign image.

    Homography works because the translation and rotation (and scale etc.) matrix
    can be combined into one operation. This matrix is called the homography
    matrix. We can then write out a set of equations
    describing the new yx, grid points as a function of the rotation and translation.
    By writing these equations for the new x and y for each grid point
    up as a matrix, we can just use singular value decomposition (SVD) to find the
    coefficients (ie the translation and rotation) and out.
    To improve this, we typically use least squares, with additional constraints
    that we now to help improve the output homography matrix.

    Most off the shell homography routines apply the rotation matrix first
    then apply the translation matrix. To reverse this ordering, set
    `self.reverse_homography` in the method `homography_on_grid_points`
    to True

    """

    def __init__(self, homography_matrix=None,
                 reverse_homography=False, reverse_shifts=False):
        """Initialisation of the Homography class
        """
        self.homography_matrix = homography_matrix
        self.reverse_homography = reverse_homography
        self.reverse_shifts = reverse_shifts

    @property
    def hm_shifts(self):
        return get_shifts_from_homography_matrix(self.homography_matrix,
                                                 self.reverse_homography,
                                                 self.reverse_shifts)

    @property
    def hm_rotation(self):
        return get_rotation_from_homography_matrix(self.homography_matrix)

    def homography_on_grid_points(self, original_xy, transformed_xy,
                                  method=transform.EuclideanTransform,
                                  reverse_homography=False,
                                  reverse_shifts=False,
                                  **kwargs):
        """Performs a fit to estimate the homography matrix that represents the grid
        transformation. Uses ransac to robustly eliminate vector outliers that
        something such as optical flow, or other feature extracting methods
        might output. The parameters are not currently changeable by the user
        since not entirely sure what are the best parameters to use and so
        are using values given in skimage tutorials.
        for Euclidean transformations (i.e., pure rotations and translations)

        Parameters
        ----------
        original_xy : nx2 array
            nx2 array of the xy coordinates of the reference grid.
        transformed_xy :  nx2 array
            nx2 array of the xy coordinates of a feature that has been transformed
            by a rotation and translation (coordinates of object in prealign grid)
        method : function, optional
            method is the what function is used to find the homography.
            The default is transform.EuclideanTransform.
        reverse_homography : bool, optional
            Homography matrix outputs the solution representing a rotation followed
            by a translation. If you want the solution to translate, then rotate
            set this to True. The default is False.
        ransac_test_values: bool
            Boolean to use the test values for ransac. If False,
            it will revert to code defined defaults values set up in the
            default_kw dictionary. Default to True.


        Returns
        -------
        self.homography_matrix : 3x3 matrix
            The Matrix that has been "performed" that resulted in the offset
            between the prealign and the reference. The inverse therefore
            describes the parameters needed to convert the prealign image
            to the reference grid.

        """
        self.reverse_homography = reverse_homography
        self.reverse_shifts = reverse_shifts

        # Get keywords from kwargs
        # If test value is True then keeping this empty to use default from ransac
        # and speed up things
        kwargs_ransac = {'min_samples': 3, 'residual_threshold': 0.5,
                      'max_trials': 1000}
        # Overwriting the keywords in case those are provided
        for key in kwargs:
            kwargs_ransac[key] = kwargs.pop(key)

        # Call to Ransac
        self.model_robust, inliers = ransac((original_xy, transformed_xy), method,
                                       **kwargs_ransac)
        self.homography_matrix = self.model_robust.params

class AlignOpticalFlow(AlignmentBase, AlignHomography):
    """Optical flow is a gradient based method that find small changes between
    two images as vectors for each pixel. Recommended for translations and
    rotations that are less than 5 pixels. If larger than this, use
    the iterative method.

    To do optical flow, the intensities between the two image much be as equal
    as possible. Changes in intensity are kinda interpreted as a scale change.
    To match the intensities, histogram equalisation, or some intensity
    scaling is needed.
    """

#    def __init__(self, prealign, reference):
#        super().__init__(prealign, reference)
#        self.homography_matrix = None
#        self.shifts = None
#        self.rotation = None

    def optical_flow(self, opf_test_values=True, **kwargs):
        """
        Performs the optical flow.

        Parameters
        ----------
        opf_test_values: bool
            Boolean to use the test values for the optical_flow. If False,
            it will revert to code defined defaults values set up in the
            default_kw dictionary. Default to True.

        Returns
        -------
        v : 2d array
            y component of the vectors describing the offset between the prealign
            and the reference image.
        u : 2d array
            x component of the vectors describing the offset between the prealign
            and the reference image.

        """
        # Get keywords from kwargs
        # If test value is True then keeping this empty to use default from ransac
        # and speed up things
        default_kw = {'attachment': 8, 'tightness': 0.12, 'num_warp': 30,
                      'num_iter': 250, 'tol': 0.0001, 'prefilter': False}

        if not opf_test_values:
            kwargs_of = default_kw

            # Overwriting the keywords in case those are provided
            for key in kwargs:
                kwargs_of[key] = kwargs.pop(key)

        else:
            kwargs_of = {}

        v, u = optical_flow_tvl1(self.reference_filter, self.prealign_filter,
                                 **kwargs_of)

        return v, u

    def _prep_arrays(self, num_per_dimension):
        """Gets a sample of coordinates from the optical flow to perform
        homography on. A sample is needed because the full image is expensive
        to run.

        Parameters
        ----------
         num_per_dimension : int
             For each axis, this sets the number of grid points to be calculated.

        Returns
        -------
        xy_stacked : num_per_dimension x2 array
            The paired xy coordinates of the vector origin.
        xy_stacked_pre : num_per_dimension x2 array
            The paired xy coordinates of the vector end.

        """
        # Vector field. V are the y component of the vectors, u are the x components
        self.v, self.u = self.optical_flow()

        y_sparse, x_sparse, v_sparse, u_sparse = \
            au.get_sparse_vector_grid(self.v, self.u, num_per_dimension)

        y_sparse_pre = y_sparse + v_sparse
        x_sparse_pre = x_sparse + u_sparse

        xy_stacked = au.column_stack_2d(x_sparse, y_sparse)
        xy_stacked_pre = au.column_stack_2d(x_sparse_pre, y_sparse_pre)

        return xy_stacked, xy_stacked_pre

    @property
    def shifts(self):
        return get_shifts_from_homography_matrix(self.matrix_transform, 
                                                 self.reverse_homography,
                                                 self.reverse_shifts)

    @property
    def rotation(self):
        return get_rotation_from_homography_matrix(self.matrix_transform)

    @property
    def total_manual_rotation(self):
        return np.sum(self.manually_applied_offsets['rotations'])

    @property
    def total_manual_translations(self):
        return np.sum(self.manually_applied_offsets['translations'], axis=0)

    def get_translation_rotation(self, num_per_dimension=50,
                                 homography_method=transform.EuclideanTransform,
                                 reverse_homography=False, 
                                 reverse_shifts=True, **kwargs):
        """Works out the translation and rotation using homography once

        Parameters
        ----------
        num_per_dimension : int, optional
             For each axis, this sets the number of grid points to be calculated.
             The default is 50.
        homography_method : function, optional
            The (skimage) transformation method. Different transforms can find
            additional types of matrix transformations that might be present.
            For alignment, we only want to find rotation and transformation. For
            only these transformations, use the Euclidean transform. The default
            is transform.EuclideanTransform.
        reverse_homography : bool, optional
            If the order that a user will use to  correct the alignment is NOT
            rotation, then translation, set this to True. The default is False.
        reverse_shifts : bool
            When True, the shifts will be multiplied by a -1 sign. Default is True.
        **kwargs: additional arguments
            Those arguments will be passed on to ransac in the homography
            calculation.

        Returns
        -------
        self.shifts : 2-array
            The recovered shifts, in pixels.
        self.rotation : float
            The recovered rotation, in degrees.
        self.homography_matrix : 3x3 matrix
            The Matrix that has been "performed" that resulted in the offset
            between the prealign and the reference. The inverse therefore
            describes the parameters needed to convert the prealign image
            to the reference grid.

        """

        reference_grid_xy, prealign_grid_xy = self._prep_arrays(num_per_dimension)

        # Using homography
        # Note: method must be a method that outputs the 3x3 homography matrix
        self.homography_on_grid_points(original_xy=reference_grid_xy,
                                       transformed_xy=prealign_grid_xy,
                                       method=homography_method,
                                       reverse_homography=reverse_homography,
                                       reverse_shifts=reverse_shifts,
                                       **kwargs)

        #updating the prealign image, this also updates `self.matrix_transform`
        self.rotate_prealign(self.hm_rotation)
        self.translate_prealign(self.hm_shifts)

    def get_iterate_translation_rotation(self, niter=1, num_per_dimension=50,
                                 homography_method=transform.EuclideanTransform,
                                 reverse_homography=False, 
                                 reverse_shifts=True, **kwargs):
        """If the solution is offset by more than ~5 pixels, optical flow struggles
        to match up the pixels. This method gets around this by applying optical
        flow multiple times to that the shifts and rotations converge to a solution.
        TODO = Make this stop after a change between solutions is smaller that
        a user defined value (i.e >1% for example)

        Parameters
        ----------
        niter : int, optional
            The number of times to perform optical flow to find the shifts and
            rotation. The default is 1, as normally we should use iterations
            within optical flow itself, and not from this stage.
        num_per_dimension : int, optional
             For each axis, this sets the number of grid points to be calculated.
             The default is 50.
        homography_method : function, optional
            The (skimage) transformation method. Different transforms can find
            additional types of matrix transformations that might be present.
            For alignment, we only want to find rotation and transformation. For
            only these transformations, use the Euclidean transform. The default
            is transform.EuclideanTransform.
        reverse_homography : bool, optional
            If the order that a user will use to  correct the alignment is NOT
            rotation, then translation, set this to True. The default is False.
        reverse_shifts : bool
            When True, the shifts will be multiplied by a -1 sign. Default is True.
        **kwargs: additional arguments
            Those arguments will be passed on to ransac in the homography
            calculation.

        Returns
        -------
        self.shifts : 2-array
            The recovered shifts, in pixels.
        self.rotation : float
            The recovered rotation, in degrees.

        """
        for i in range(niter):
            # This finds
            self.get_translation_rotation(num_per_dimension, homography_method,
                                          reverse_homography, reverse_shifts,
                                          **kwargs)


class AlignmentCrossCorrelate(AlignmentBase):
    """Finds the translational offset between the reference and prealign images
    using phase cross correlation. Phase cross correlation works by converting
    the images into fourier space. Here, translational offsets are represented
    as a phase difference, where there is no signal when they are out of phase,
    and a sharp peak when the image is in phase. This sharp peak is used
    to work out the offset between two images.
    """

    def fft_phase_correlation(self, prealign, reference, resolution=1000, **kwargs):
        """
        Method for performing the cross correlation between two images.
        Parameters
        ----------
        prealign : 2d array
            Image to be aligned.
        reference : 2d array
            Reference image with the correct alignment.
        resolution : int, optional
            Determines how precise the algorthm will run. For example, 10 will
            find a solution closest to the first decimal place, 100 the second
            etc. The default is 1000.
        **kwargs : skimage.registration.phase_cross_correlation properties, optional
            kwargs are optional parameters for
            skimage.registration.phase_cross_correlation
        Returns
        -------
        self.shifts : 2-array
            The offset needed to align the prealign image with the
            reference image.  Positive values shift the prealign in the negative
            direction i.e A coordinate of 3,3, with an offset of 1,1
            will be shifted to 2,2. If the image have been prepared properly
            the returned shifts encode rotational offsets and scale changes
            between the two images.
        """
        upsample_factor = kwargs.pop('upsample_factor', resolution)
        shifts, error, phasediff = phase_cross_correlation(
            reference_image=reference,
            moving_image=prealign,
            upsample_factor=upsample_factor, **kwargs
            )
        return shifts
