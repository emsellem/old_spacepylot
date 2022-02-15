# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:50:06 2021

@author: Liz_J
"""
__author__ = "Elizabeth Watkins"
__copyright__ = "Elizabeth Watkins"
__license__   = "MIT License"
__contact__ = "<liz@email"

import numpy as np

from .params import pcc_params


class VerbosePrints:
    """
    General print statements to inform the user what the alignment algorithms
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
            print('\nRemoving boundary of %d. Filtering image using '
                  'histogram equalisation and high pass filter %s with arguments:'
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
            A coordinate of 3,3, with an offset of 1,1 will be shifted to 2,2.

        Returns
        -------
        None.

        """
        if self.verbose and np.sum(yx_offset) != 0:
            print('\nAdditional offset of:\ny=%.4f, x=%4f pixels has been applied.'
                  ' All manually given offsets recorded in'
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

        if self.verbose and rotation_angle != 0:
            print('\nAdditional rotation of: theta=%4f degrees '
                  'has been applied. All manually given offsets recorded in'
                  ' `self.manually_applied_offsets`' % rotation_angle)

    def get_translation(self, shifts, added_shifts=(0, 0)):
        """
        Prints the translation found by the align object

        Parameters
        ----------
        shifts : 2-array
            The y and x offset needed to align the prealign image with the
            reference image.  Positive values shift the prealign in the negative direction i.e
            A coordinate of 3,3, with an offset of 1,1 will be shifted to 2,2.
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

