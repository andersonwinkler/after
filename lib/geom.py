#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 18:27:11 2025

@author: winkler
"""

import numpy as np

# =============================================================================
def line_intersection(A,B,C,D):
    '''
    Compute the coordinates of the intersection between line segments AB and CD.
    Returns also parameters s and t. If these are between 0 and 1, the intersection
    is, respectively, within AB and CD.
    Implementation based on:
        * Vince J. Foundation Mathematics for Computer Science.
          4th ed. Springer, 2024. Chapter 20, pp 417-419.

    Parameters
    ----------
    A : NumPy array (N by 3)
        Cartesian coordinates of point A.
    B : NumPy array (N by 3)
        Cartesian coordinates of point B.
    C : NumPy array (N by 3)
        Cartesian coordinates of point C.
    D : NumPy array (N by 3)
        Cartesian coordinates of point D.

    Returns
    -------
    P : NumPy array (N by 3)
        Cartesian coordinates of intersection point P.
    '''
    # Find the parameters s and t via least squares. This avoids having to figure
    # out if there are dependent equations
    N    = A.shape[0]
    dat  = np.concatenate((A[:,0]-C[:,0],A[:,1]-C[:,1],A[:,2]-C[:,2])).reshape((N,3), order='F')
    scol = np.concatenate((D[:,0]-C[:,0],D[:,1]-C[:,1],D[:,2]-C[:,2])).reshape((N,3), order='F')
    tcol = np.concatenate((A[:,0]-B[:,0],A[:,1]-B[:,1],A[:,2]-B[:,2])).reshape((N,3), order='F')
    des  = np.concatenate((scol[:,:,None], tcol[:,:,None]), axis=2)
    st   = np.zeros((N,2)) # for the terms s and t of the parameterization
    rnk  = np.zeros((N))
    for p in range(A.shape[0]):
        st[p], _, rnk[p], _ = np.linalg.lstsq(des[p,:,:], dat[p,:], rcond=None)
    
    # Mark parallel lines with None
    st[rnk < 2] = None
    
    # Intersection point
    # It can be computed using A, B and t, or using C, D, and s
    # Results should be the same
    #Pt = A + st[:,1,None]*(B-A)
    #Ps = C + st[:,0,None]*(D-C)
    P   = C + st[:,0,None]*(D-C)
    return P, st[:,0], st[:,1]

# =============================================================================
def normal2zaxis(n):
    '''
    Compute a 3x3 rotation matrix that changes the coordinate system such
    that n @ rot is a new coordinate system with the z-axis along n.

    Parameters
    ----------
    n : Input vector

    Returns
    -------
    rot : Rotation matrix
    '''
    n     /= np.linalg.norm(n)
    z      = np.array([0,0,1])
    axis   = np.cross(z,n)
    naxis  = np.linalg.norm(axis)
    if naxis > 0:
        axis  /= naxis
    angle  = np.arccos(np.clip(np.dot(z,n),-1,1))
    K = np.array([
        [   0 ,    -axis[2],  axis[1]],
        [ axis[2],       0 , -axis[0]],
        [-axis[1],  axis[0],       0]])
    rot    = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return rot