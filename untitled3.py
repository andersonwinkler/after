#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 17:53:42 2025

@author: winkler
"""

import numpy as np
from lib import io
from lib import geom
from lib import surf

vtx7, fac7, info, stamp = io.read_surf('/opt/freesurfer/8.0.0/subjects/fsaverage/surf/lh.sphere.reg')
vtx2, fac2 = geom.icodown(vtx7, fac7, 5)
vtx1, fac1 = geom.icodown(vtx2, fac2, 1)
vtx0, fac0 = geom.icodown(vtx1, fac1, 1)

#dpf7 = np.zeros(fac7.shape[0]) + 1
#dpf1 = np.zeros(fac1.shape[0]) + 1
#dpf0 = geom.dpxdown(dpf7, 7, vtx=vtx7, fac=fac7, fsmode=False, pycno=None)

D = surf.fractal_dimension(vtx1, fac1, fsmode=True)

dpv1 = geom.dpf2dpv(D, fac0, pycno=False)
dpv2 = geom.dpf2dpv(D, fac0, facu=fac1, pycno=False, fsmode=True)
print(np.sum(D))
print(np.sum(dpv1))
print(np.sum(dpv2))