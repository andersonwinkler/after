#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:11:57 2024

@author: winkleram
"""

import os
import numpy as np
import argparse
import time
from lib.io    import read_surf, write_surf
from lib.utils import progress_bar

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(
                        description=\
                        'Retessellate surfaces to a common grid. ' + \
                        'Invoke either with --subj (and optionally --subjdir and --srf)' + \
                        'or with --srfsph, --trgsph, --srcsrf, and --trgsrf.' + \
                        'In the latter case, the paths to the files must be specified.',
                        epilog=\
                        'Anderson M. Winkler / UTRGV / May 2025 / https://brainder.org')
    # Options to call specifying a subject
    parser.add_argument('--subj',
                        help='List of subjects separated by commas',
                        type=str, required=False, default=None)
    parser.add_argument('--subjdir',
                        help='Subjects directory (usually SUBJECTS_DIR)',
                        type=str, required=False, default=None)
    parser.add_argument('--srf',
                        help='Surface to retessellate, if invoked with "--subj" (default: orig)',
                        type=str, required=False, default='orig')
    # Options to call with specific files
    parser.add_argument('--srcsph',
                        help='Source sphere (typically subj/surf/?h.sphere.reg)',
                        type=str, required=False, default=None)
    parser.add_argument('--trgsph',
                        help='Target sphere (typically fsaverage/surf/?h.sphere.reg)',
                        type=str, required=False, default=None)
    parser.add_argument('--srcsrf',
                        help='Source surface (typically subj/surf/?h.white)',
                        type=str, required=False, default=None)
    parser.add_argument('--trgsrf',
                        help='Target surface (typically subj/after/?h.white.retess), to be created',
                        type=str, required=False, default=None)
    # Show a nice progress bar (only use in interactive sessions)
    parser.add_argument('--progress',
                        help='Show a progress bar',
                        action='store_true', required=False, default=False)

    # Parsing proper
    args     = parser.parse_args()
    subj     = args.subj
    subjdir  = args.subjdir
    srf      = args.srf
    srcsph   = args.srcsph
    trgsph   = args.trgsph
    srcsrf   = args.srcsrf
    trgsrf   = args.trgsrf
    progress = args.progress
    
    # Check whether we are using a default FS subject structure, or isolated, custom files
    customargs = [srcsph, trgsph, srcsrf, trgsrf]
    if subj is None and None in customargs:
        raise SyntaxError('Either "--subj" or all of "--srcsph", "--trgsph", "--srcsrf" and "--trgsrf" must be provided.')
    if subj is not None and not None in customargs:
        raise SyntaxError('Cannot use "--subj" together with "--srcsph", "--trgsph", "--srcsrf" or "--trgsrf".')
    
    # If a list of subjects was provided, let's use it
    if subj is not None:
        # Find SUBJECTS_DIR
        if subjdir is None:
            subjdir = os.getenv('SUBJECTS_DIR')
        if subjdir == '' or subjdir is None:
            raise SyntaxError('Either --subjdir option must be provided, or the environmental variable SUBJECTS_DIR must be set')
        
        # Find fsaverage
        fshome = os.getenv('FREESURFER_HOME')
        if os.path.exists(os.path.join(subjdir, 'fsaverage')):
            fsavgdir = os.path.join(subjdir, 'fsaverage')
        elif fshome is not None and fshome != '' and os.path.exists(os.path.join(fshome, 'subjects', 'fsaverage')):
            fsavgdir = os.path.join(fshome, 'subjects', 'fsaverage')
        else:
            raise FileNotFoundError('"fsaverage" not found; make sure it exists in the subjects directory or that FreeSurfer is correctly installed')
    
        # Make a list of subjects
        subjlist = subj.split(',')   
        H = ['lh', 'rh']
    else:
        subjlist = [None]
        H = [None]
        
    # For each subject and each hemisphere
    for subj in subjlist:
        for h in H:
            if subj is None or h is None:
                path1 = srcsph
                path2 = trgsph
                path3 = srcsrf
                path4 = trgsrf
            else:
                os.makedirs(os.path.join(subjdir, subj, 'after'), exist_ok=True)
                path1 = os.path.join(subjdir, subj, 'surf',  '{}.sphere.reg'.format(h))
                path2 = os.path.join(fsavgdir,      'surf',  '{}.sphere.reg'.format(h))
                path3 = os.path.join(subjdir, subj, 'surf',  '{}.{}'.format(h, srf))
                path4 = os.path.join(subjdir, subj, 'after', '{}.{}.retess'.format(h, srf))
            print('Retessellating {}'.format(path3))
            
            # Default margin
            marg =  0.05;
            
            # Check if the first three paths exist.
            if not os.path.exists(path1):
                raise FileNotFoundError('File does not exist: {}'.format(path1))
            if not os.path.exists(path2):
                raise FileNotFoundError('File does not exist: {}'.format(path2))
            if not os.path.exists(path3):
                raise FileNotFoundError('File does not exist: {}'.format(path3))
        
            # Load the surfaces using nibabel for FreeSurfer formats
            # Will use the vertex coordinates as stored, which is equivalent to
            # converting to RAS in this case. For other operations that require
            # the RAS coordinates, these can be obtained with vtx = vtx + info['cras']
            vtx1, fac1, _, _ = read_surf(path1)
            vtx2, fac2, _, _ = read_surf(path2)
            vtx3, fac3, info3, stamp3 = read_surf(path3)
            nF1 = fac1.shape[0]
            nV2 = vtx2.shape[0]
            
            # Sanity check
            if not np.array_equal(fac1, fac3):
                raise ValueError("The source surface and the surface to be retessellated must have the same geometry.")
        
            # Where the result is going to be stored
            vtx4 = np.zeros((nV2, 3))
            
            # Vertices' coords per face
            facvtx1 = np.hstack((vtx1[fac1[:,0],:], vtx1[fac1[:,1],:], vtx1[fac1[:,2],:]))
            
            # Face barycenter
            xbary = np.mean(facvtx1[:, [0, 3, 6], None], axis=1)    # x-coordinate
            ybary = np.mean(facvtx1[:, [1, 4, 7], None], axis=1)    # y-coordinate
            zbary = np.mean(facvtx1[:, [2, 5, 8], None], axis=1)    # z-coordinate
            cbary = np.hstack((xbary, ybary, zbary))                # Cartesian coordinates of the barycenters
            r     = np.sqrt(xbary**2 + ybary**2 + zbary**2)         # radius
            theta = np.arctan2(ybary, xbary)                        # azimuth (angle in x-y plane)
            phi   = np.arctan2(zbary, np.sqrt(xbary**2 + ybary**2)) # elevation (angle from z-axis)
            sbary = np.hstack((theta, phi, r))                      # Spherical coordinates of the barycenters
            
            # Pre-calculated sines and cosines of azimuth and elevation:
            sinA = np.sin(sbary[:,0,None])
            sinE = np.sin(sbary[:,1,None])
            cosA = np.cos(sbary[:,0,None])
            cosE = np.cos(sbary[:,1,None])
            
            # Pre-calculated rotation matrices
            rotM = np.column_stack((cosA * cosE, sinA*cosE, sinE, -sinA, cosA, np.zeros(nF1), -cosA*sinE, -sinA*sinE, cosE))
            
            # Random angle around X
            rndangX = np.random.rand() * np.pi
            rndangX = np.pi/3
            sinX = np.sin(rndangX)
            cosX = np.cos(rndangX)
            rotM = np.column_stack((
                rotM[:,:3],  
                rotM[:,3] * cosX + rotM[:,6] * sinX,
                rotM[:,4] * cosX + rotM[:,7] * sinX,
                rotM[:,5] * cosX + rotM[:,8] * sinX,
                rotM[:,6] * cosX - rotM[:,3] * sinX,
                rotM[:,7] * cosX - rotM[:,4] * sinX,
                rotM[:,8] * cosX - rotM[:,5] * sinX 
            ))
            
            # Pre-calculated min and max for each face and bounding box
            minF = np.column_stack(( np.min(facvtx1[:, [0, 3, 6]], axis=1), np.min(facvtx1[:, [1, 4, 7]], axis=1), np.min(facvtx1[:, [2, 5, 8]], axis=1) ))
            maxF = np.column_stack(( np.max(facvtx1[:, [0, 3, 6]], axis=1), np.max(facvtx1[:, [1, 4, 7]], axis=1), np.max(facvtx1[:, [2, 5, 8]], axis=1) ))
            b    = np.tile(np.max((maxF - minF), axis=1).reshape(-1, 1), (1, 3)) * marg
            minF = minF - b
            maxF = maxF + b
        
            # For each source face
            start_time = time.time()
            for f in range(nF1):
                if progress:
                    progress_bar(f, nF1, start_time, prefix='Processing faces:', min_update_interval=1)
                
                vidx = fac1[f, :]         # Indices of the vertices for face f
                Fvtx = vtx1[vidx, :]      # Corresponding vertex coordinates from vtx1
                
                # Candidate vertices
                Cidx = np.all((vtx2 >= minF[f, :]) & (vtx2 <= maxF[f, :]), axis=1)  # Logical condition across columns
                Cvtx = vtx2[Cidx, :]  # Extract candidate vertices
                Cidxi = np.where(Cidx)[0]  # Indices of the candidate vertices
                
                # Concatenate the face vertices and candidate vertices
                Avtx = np.vstack((Fvtx, Cvtx)) @ rotM[f, :].reshape(3, 3).T
        
                # Convert to azimuthal gnomonic
                Gvtx = np.ones_like(Avtx)  # The 3rd col will remain full of ones
                Gvtx[:, 0] = Avtx[:, 1] / Avtx[:, 0]  # Tangent of the angle on the XY plane
                Gvtx[:, 1] = Avtx[:, 2] / Avtx[:, 0]  # Tangent of the angle on the XZ plane
                T = Gvtx[:3, :] # Face coords for the test below
                aT = np.linalg.det(T) # Face total area (2x the area, actually)
                
                # For every candidate vertex
                for v in range(len(Cidxi)):
                    
                    # Compute the areas for the subtriangles (2x area actually)
                    # Subtriangle A
                    tA = T.copy()  # Copy the original T
                    tA[0, :] = Gvtx[v + 3, :]
                    aA = abs(np.linalg.det(tA))
                    
                    # Subtriangle B
                    tB = T.copy()
                    tB[1, :] = Gvtx[v + 3, :]
                    aB = abs(np.linalg.det(tB))
                    
                    # Subtriangle C
                    tC = T.copy()
                    tC[2, :] = Gvtx[v + 3, :]
                    aC = abs(np.linalg.det(tC))
                    
                    # Test if the point is inside the face
                    if np.float32(aT) == np.float32(aA + aB + aC): # Use float32 (single) to emulate Matlab. However, np.isclose would have been more pythonic
                    
                        # Weight by the areas and interpolate the value between the 3 vertices
                        vtx4[Cidxi[v], :] = np.dot([aA, aB, aC], vtx3[vidx, :]) / aT
                
            # Save output surface
            print('Saving results to: {}'.format(path4))
            write_surf(path4, vtx4, fac2, info3, stamp=True)
    print('Finished retessellation.')