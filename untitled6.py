#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 12:53:43 2025

@author: winkler
"""

import lib
import numpy as np
import time

nmax = 5
t = np.zeros(nmax)
vtx7, fac7, info, stamp = lib.io.read_surf('/opt/freesurfer/8.0.0/subjects/fsaverage/surf/lh.white')

for ic in range(nmax):
    # Read the surface data
    vtx, fac = lib.platonic.icodown(vtx7, fac7, 7-ic)
    
    start_time = time.time()
    dist_ma = lib.surf.geodesic_distances(vtx, fac, method='mitchell', iterate_vtx=False)
    end_time = time.time()
    t[ic] = end_time - start_time
    print(f"Time taken {ic}: {t[ic]} seconds (Mitchell-all)")

    start_time = time.time()
    dist_mv = lib.surf.geodesic_distances(vtx, fac, method='mitchell', iterate_vtx=True)
    end_time = time.time()
    t[ic] = end_time - start_time
    print(f"Time taken {ic}: {t[ic]} seconds (Mitchell-iterate)")

    start_time = time.time()
    dist_c = lib.surf.geodesic_distances(vtx, fac, method='crane')
    end_time = time.time()
    t[ic] = end_time - start_time
    print(f"Time taken {ic}: {t[ic]} seconds (Crane)")
    
    start_time = time.time()
    dist_d = lib.surf.geodesic_distances(vtx, fac, method='dijkstra')
    end_time = time.time()
    t[ic] = end_time - start_time
    print(f"Time taken {ic}: {t[ic]} seconds (Dijkstra)")