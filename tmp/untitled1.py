#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 20:47:30 2025

@author: winkler
"""
import numpy as np
from lib import io
from lib import geom

def edgedist(vtx1,vtx2):
    D = np.sqrt( \
        (vtx1[:,0,None]-vtx2[:,0,None].T)**2 + \
        (vtx1[:,1,None]-vtx2[:,1,None].T)**2 + \
        (vtx1[:,2,None]-vtx2[:,2,None].T)**2)
    return D

vtx7, fac7, info, stamp = io.read_surf('/opt/freesurfer/8.0.0/subjects/fsaverage/surf/lh.sphere.reg')
vtx1, fac1 = geom.icodown(vtx7, fac7, 1)
vtx0, fac0 = geom.icodown(vtx7, fac7, 0)


vtxmy0, facmy0 = geom.icosahedron(meas='cr', value=100, fsmode=True)
vtxmy1, facmy1 = geom.icoup(vtxmy0, facmy0, 1, fsmode=True)
#io.write_surf('/home/winkler/lh.sphere1x.reg', vtx0x, fac0x, info)


# A = np.array([9,7,7,3,8,5,7,7,8,9])/10
# U, uidx, uinv = np.unique(A, axis=0, return_index=True, return_inverse=True)
# sidx = np.sort(uidx)
# Agood = A[sidx]

# aidx = np.argsort(np.argsort(uidx))
# sinv = aidx[uinv]

# F = np.arange(len(A))
# Fgood = sinv[F]

