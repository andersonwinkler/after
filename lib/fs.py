#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 14:40:30 2025

@author: winkler
"""
import os
from . import io
from . import geom

def make_fine_fsaverage(outdir=None):
    '''
    Make fsaverage surfaces (icosahedrons recursively subdivided) and save them
    in the module's tree structure. These are "finer" than the original shipped
    with FreeSurfer in that vertex coordinates have more decimal places, and
    recursions span all range 0-9, as opposed to only 3-7.
    Instead of lh and rh (which are identical), saves as "xh".
    '''
    
    # Find FreeSurfer
    fshome = os.getenv('FREESURFER_HOME')
    if fshome is None:
        raise ValueError('Error: FREESURFER_HOME variable not set. Stopping.')
    
    # Where to save
    if outdir is not None:
        outdir = os.path.join(os.path.dirname(__file__), 'etc', 'fsaverage')
    
    # Read lh.sphere.reg so we can get the coordinate transformation (just in case)
    # The lh.sphere.reg is binarily identical to rh.sphere.reg, so just one of these is sufficient
    _, _, info, _ = io.read_surf(os.path.join(fshome,'subjects','fsaverage','surf','lh.sphere.reg'))
    
    # Create and save fsaverage0 (no subdivision)
    print('Working on subdivision 0')
    vtx, fac = geom.icosahedron(meas='cr', value=100, fsmode=True)
    io.write_surf(os.path.join(outdir, 'xh.fsaverage0.fine.sphere.reg'), vtx, fac, info)
    
    # Recursively subdivide and save
    for n in range(1,10):
        print(f'Working on subdivision {n}')
        vtx, fac = geom.icoup(vtx, fac, 1, fsmode=True)
        io.write_surf(os.path.join(outdir, f'xh.fsaverage{n}.fine.sphere.reg'), vtx, fac, info)
