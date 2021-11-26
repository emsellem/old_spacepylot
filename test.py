# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 21:03:10 2021

@author: Liz_J
"""

import autoAlignment.alignment as align

ngc = '1672'

GALAXY = 'n' + ngc
PATH = 'C:\\Users\\Liz_J\\Documents\\muse_new_align_test\\%s\\' % GALAXY

ref_name = 'NGC%s_Rc_flux_nosky-matched_rpj_0001.fits'
ref_path = PATH + '\\WFI_BB\\wfi_bb_ref\\P01\\' + ref_name % ngc

pre_name = 'simplified_hdu_0001.fits'
prealign_path = PATH + 'WFI_BB\\simplified_hdu\\P01\\' + pre_name
trys = 10
shifts_op = [0,0]
rotation_op = 0


op = align.Align_optical_flow.from_fits(prealign_path, ref_path, guess_translation=shifts_op, guess_rotation=rotation_op, convolve_prealign=None)

shifts_op, rotation_op = op.get_translation_rotation()

op_plot = AlignmentPlotting.from_align_object(op)
plt.clf()
op_plot.red_blue_before_after()


