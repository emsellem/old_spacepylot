# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 21:03:10 2021

@author: Liz_J
"""

import astroAlign.alignment as align
import astroAlign.plotting as pl


ref_path = 
prealign_path =
trys = 10
shifts_op = [0,0]
rotation_op = 0

convolve_prealign = 1

op = align.Align_optical_flow.from_fits(prealign_path, ref_path, guess_translation=shifts_op, guess_rotation=rotation_op, convolve_prealign=convolve_prealign, verbose=True)
shifts_op, rotation_op = op.get_translation_rotation(itera=1)

op_plot = pl.AlignmentPlotting.from_align_object(op)

op_plot.red_blue_before_after()
op_plot.illistrate_vector_fields()


