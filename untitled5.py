#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 12:53:43 2025

@author: winkler
"""

import lib
import gdist
import potpourri3d as pp3d
import numpy as np
import time

nmax = 1
t = np.zeros(nmax)
vtx7, fac7, info, stamp = lib.io.read_surf('/opt/freesurfer/8.0.0/subjects/fsaverage/surf/lh.sphere.reg')

for ic in range(nmax):
    # Read the surface data
    vtx, fac = lib.platonic.icodown(vtx7, fac7, 7-ic)
    nV = vtx.shape[0]
    
    start_time = time.time()
    solver = pp3d.MeshHeatMethodDistanceSolver(vtx, fac, t_coef=1., use_robust=True)
    pdpp3    = np.zeros((nV,nV))
    for v in range(nV):
        pdpp3[v] = solver.compute_distance(v)
    end_time = time.time()
    t[ic] = end_time - start_time
    print(f"Time taken {ic}: {t[ic]} seconds")
    
    # Convert to native byte order and ensure exact data types expected by gdist
    vtx_native = vtx.astype(np.float64, order='C', copy=False)
    fac_native = fac.astype(np.int32,   order='C', copy=False)
    
    # Ensure contiguous arrays with correct byte order
    vtx = np.ascontiguousarray(vtx_native, dtype=np.float64)
    fac = np.ascontiguousarray(fac_native, dtype=np.int32)
    
    # Compute geodesic distances
    start_time = time.time()
    #pdgeo = gdist.local_gdist_matrix(vtx, fac, max_distance=gdist.numpy.inf).toarray()
    xx = gdist.compute_gdist(vtx, fac, source_indices=np.array([11], dtype=np.int32), max_distance=gdist.numpy.inf)
    end_time = time.time()
    t[ic] = end_time - start_time
    print(f"Time taken {ic}: {t[ic]} seconds")


