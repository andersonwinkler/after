#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 22:36:30 2025

@author: winkleram
"""


import os
import argparse
import numpy as np
import time
from lib.io import read_surf, read_obj, write_curv, write_mgh
from lib.utils import progress_bar

def calc_normals(vtx, fac): # =================================================
    '''
    Compute vertex and face normals.
    Vertex normals use the Max et al. (1999) algorithm.
    
    Parameters
    ----------
    vtx : NumPy array with 3 columns (float).
        Vextex coordinates.
    fac : NumPy array with 3 columns (int).
        Face indices.

    Returns
    -------
    vn : NumPy array with 3 columns (float).
        Vertex normals.
    fn : NumPy array with 3 columns (float).
        Face normals.
    '''
    
    # Number of vertices
    nvtx = vtx.shape[0]
        
    # Vertices and edges of each face, their lengths
    tri     = vtx[fac]
    edges   = tri[:,[2,0,1],:] - tri[:,[1,2,0],:] # a=C-B, b=A-C, c=B-A
    lengths = np.linalg.norm(edges, axis=2)

    # Face normal
    area, fn = signed_area(tri)

    # Weights -- Simplified, faster, same results as Max (1999)
    cA = area / (lengths[:,1] * lengths[:,2]) ** 2 # (b*c)^2
    cB = area / (lengths[:,2] * lengths[:,0]) ** 2 # (c*a)^2
    cC = area / (lengths[:,0] * lengths[:,1]) ** 2 # (a*b)^2

    # Accumulate weighted face normal contributions
    # thus avoiding to iterate over vertices
    vn = np.zeros((nvtx,3))
    np.add.at(vn, fac[:,0], cA[:,None] * fn)
    np.add.at(vn, fac[:,1], cB[:,None] * fn)
    np.add.at(vn, fac[:,2], cC[:,None] * fn)

    # Normalize normals to unit.
    # You may opt to comment out to take into account transitions
    # e.g., from positive to negative.
    vn  /= np.linalg.norm(vn, axis=1)[:,None]
    fn  /= np.linalg.norm(fn, axis=1)[:,None]
    return vn, fn

def signed_area(tri, rfn=None): # =============================================
    '''
    Compute the signed area of a triangle in relation to a reference normal.
    If the normals are parallel, area is positive; if antiparallel, negative.
    This is an auxiliary function for the computation of the Voronoi areas.

    Parameters
    ----------
    tri : NumPy array, num faces by 3 by 3
        Array of vertex coordinates of each face.
        First dimension has the face indices (0..numfaces)
        Second dimension has the vertices ABC
        Third dimension has the coordinates (x,y,z)
    rfn : NumPy array, optional
        Reference normal for each face in tri. The default is None.

    Returns
    -------
    area : NumPy array num faces by 1
        Signed area of each face
    fn : NumPy array num faces by 3
        Face normals (x,y,z).

    '''
    edges = tri[:,[2,0,1],:] - tri[:,[1,2,0],:] # a=C-B, b=A-C, c=B-A
    fn    = np.cross(edges[:,1,:], edges[:,2,:], axis=1) # b and c
    area  = np.linalg.norm(fn, axis=1) / 2.0
    idx   = area != 0
    fn[idx,:] /= np.linalg.norm(fn[idx,:], axis=1)[:,None]
    if rfn is not None:
        sgn   = np.sign(np.sum(fn * rfn, axis=1))
        area *= sgn
    return area, fn

def line_intersection(A,B,C,D): # =============================================
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

def calc_voronoi_areas(vtx, fac): # ===========================================
    '''
    Compute the Voronoi areas for each vertex and, within a face, for each
    vertex of that face.

    Parameters
    ----------
    vtx : NumPy array with 3 columns (float).
        Vextex coordinates.
    fac : NumPy array with 3 columns (int).
        Face indices.

    Returns
    -------
    vorv : NumPy vector (float)
        Voronoi area per vertex.
    vorf : NumPy array with 3 columns (float).
        Voronoi area per vertex per face.
    areas : NumPy vector (float)
        Area per face

    Note that np.sum(vorf, axis=1) is the same as area, whereas vorv contains
    the sum of all values in vorf for that vertex (referenced by fac).
    '''
    
    # Number of vertices and faces
    nvtx = vtx.shape[0]
    nfac = fac.shape[0]
    
    # Vertex coordinates of each original triangular face
    triABC = vtx[fac] # dim0: face indices; dim1: vertices ABC; dim2: (x,y,z) coords.
    
    # Vertex coordinates of the midpoints of the edges abc, opposite to
    # the vertices ABC, respectively.
    # These don't form useful triangles themselves but will be used a lot later
    triabc = (triABC[:,[1,2,0],:] + triABC[:,[2,0,1],:])/2.0
    
    # Vertex coordinates of circumcenter
    eA   = triABC[:,1,:] - triABC[:,2,:] # edge opposite to A, centered at C
    eB   = triABC[:,0,:] - triABC[:,2,:] # edge opposite to B, centered at C
    cp = np.cross(eA,eB)
    Pnum = np.cross((np.linalg.norm(eA, axis=1, keepdims=True)**2*eB - 
                     np.linalg.norm(eB, axis=1, keepdims=True)**2*eA), cp)
    Pden = 2*np.linalg.norm(cp, axis=1, keepdims=True)**2
    P    = Pnum / Pden + triABC[:,2,:]
    P    = P[:,None,:]
    
    # Voronoi subtriangles (two per Voronoi area)
    tri = {}
    # For vertex A, these are AcP and APb
    tri['AcP'] = np.concatenate((triABC[:,0,None,:], triabc[:,2,None,:], P), axis=1)
    tri['APb'] = np.concatenate((triABC[:,0,None,:], P, triabc[:,1,None,:]), axis=1)
    
    # For vertex B, these are BaP and BPc
    tri['BaP'] = np.concatenate((triABC[:,1,None,:], triabc[:,0,None,:], P), axis=1)
    tri['BPc'] = np.concatenate((triABC[:,1,None,:], P, triabc[:,2,None,:]), axis=1)
    
    # For vertex C, these are CbP and CPa
    tri['CbP'] = np.concatenate((triABC[:,2,None,:], triabc[:,1,None,:], P), axis=1)
    tri['CPa'] = np.concatenate((triABC[:,2,None,:], P, triabc[:,0,None,:]), axis=1)
    
    # Area and reference normal of the main (original) triangle ABC
    areaABC, fnABC = signed_area(triABC)
    
    # Compute the signed areas of each Vornoi subtriangle; sign is in relation
    # to the reference normal, i.e., the normal of the main triangle ABC.
    area = {}
    for key in tri:
        area[key] = signed_area(tri[key], fnABC)[0]
    
    # When there are negative areas, the triangles are obtuse, and simply adding
    # the signed areas of the Voronoi subtriangles still doesn't give the correct
    # result, i.e., the Voronoi area of the obtuse vertex ends up overestimated,
    # whereas the Voronoi areas of the two acute vertices end up underestimated.
    # Here we compute the areas of the respective subtriangles to test where they
    # are negative, the correct the under/overestimation. All areas are signed
    # Initialize variables for later
    for key in ['BmP','CPn','CmP','APn','AmP','BPn']:
        tri[key] = np.full((nfac,3,3), 0.0)
    m = np.full(P.shape, 0.0)
    n = np.full(P.shape, 0.0)
    
    # Areas of auxiliary subtriangles; we could do without them but they make the code clearer
    area['ABP'] = area['AcP'] + area['BPc']
    area['APC'] = area['CbP'] + area['APb']
    area['PBC'] = area['BaP'] + area['CPa']
    
    # If the circumcenter P is outside edge c (segment AB)
    idx = area['ABP'] < 0
    if idx.any():
        m[idx,:,:] = line_intersection(triABC[idx,0,:], triABC[idx,1,:], triabc[idx,1,:], P[idx,0,:])[0][:,None,:] # intersection AB x bP
        n[idx,:,:] = line_intersection(triABC[idx,0,:], triABC[idx,1,:], triabc[idx,0,:], P[idx,0,:])[0][:,None,:] # intersection AB x aP
        tri['AmP'][idx,:,:] = np.concatenate((triABC[idx,0,None,:], m[idx,:,:], P[idx,:,:]), axis=1)
        tri['BPn'][idx,:,:] = np.concatenate((triABC[idx,1,None,:], P[idx,:,:], n[idx,:,:]), axis=1)
    
    # If the circumcenter P is outside edge b (segment CA)
    idx = area['APC'] < 0
    if idx.any():
        m[idx,:,:] = line_intersection(triABC[idx,2,:], triABC[idx,0,:], triabc[idx,0,:], P[idx,0,:])[0][:,None,:] # intersection CA x aP
        n[idx,:,:] = line_intersection(triABC[idx,2,:], triABC[idx,0,:], triabc[idx,2,:], P[idx,0,:])[0][:,None,:] # intersection CA x cP
        tri['CmP'][idx,:,:] = np.concatenate((triABC[idx,2,None,:], m[idx,:,:], P[idx,:,:]), axis=1)
        tri['APn'][idx,:,:] = np.concatenate((triABC[idx,0,None,:], P[idx,:,:], n[idx,:,:]), axis=1)
    
    # If the circumcenter P is outside edge a (segment BC)
    idx = area['PBC'] < 0
    if idx.any():
        m[idx,:,:] = line_intersection(triABC[idx,1,:], triABC[idx,2,:], triabc[idx,2,:], P[idx,0,:])[0][:,None,:] # intersection BC x cP
        n[idx,:,:] = line_intersection(triABC[idx,1,:], triABC[idx,2,:], triabc[idx,1,:], P[idx,0,:])[0][:,None,:] # intersection BC x bP
        tri['BmP'][idx,:,:] = np.concatenate((triABC[idx,1,None,:], m[idx,:,:], P[idx,:,:]), axis=1)
        tri['CPn'][idx,:,:] = np.concatenate((triABC[idx,2,None,:], P[idx,:,:], n[idx,:,:]), axis=1)
    
    # Compute the areas for these subtriangles.
    # We could compute just for the relevant triangles but this computation is fast
    # anyway even for large meshes, so let's keep the code cleaner and clearer
    for key in ['BmP','CPn','CmP','APn','AmP','BPn']:
        area[key] = signed_area(tri[key], fnABC)[0]
    
    # These are the excesses or shortages, to be added or subtracted later.
    # Note that the area is always signed.
    area['Pmc'] = area['AcP'] - area['AmP'] # refer to case ABP being negative
    area['Pcn'] = area['BPc'] - area['BPn'] # refer to case ABP being negative
    area['Pmb'] = area['CbP'] - area['CmP'] # refer to case APC being negative
    area['Pbn'] = area['APb'] - area['APn'] # refer to case APC being negative
    area['Pma'] = area['BaP'] - area['BmP'] # refer to case PBC being negative
    area['Pan'] = area['CPa'] - area['CPn'] # refer to case PBC being negative
    
    # Now apply the correction where needed. Recall the areas are always signed.
    # If the circumcenter P is outside edge c (segment AB)
    idx = area['ABP'] < 0
    if idx.any():
        area['AcP'][idx] -= area['Pmc'][idx]
        area['BPc'][idx] -= area['Pcn'][idx]
        area['CbP'][idx] += area['Pmc'][idx] # This isn't accurate but will be correct when added to area['CPa']
        area['CPa'][idx] += area['Pcn'][idx] # This isn't accurate but will be correct when added to area['CbP']
        
    # If the circumcenter P is outside edge b (segment CA)
    idx = area['APC'] < 0
    if idx.any():
        area['CbP'][idx] -= area['Pmb'][idx]
        area['APb'][idx] -= area['Pbn'][idx]
        area['BaP'][idx] += area['Pmb'][idx] # This isn't accurate but will be correct when added to area['BPc']
        area['BPc'][idx] += area['Pbn'][idx] # This isn't accurate but will be correct when added to area['BaP']
    
    # If the circumcenter P is outside edge a (segment BC)
    idx = area['PBC'] < 0
    if idx.any():
        area['BaP'][idx] -= area['Pma'][idx]
        area['CPa'][idx] -= area['Pan'][idx]
        area['AcP'][idx] += area['Pma'][idx] # This isn't accurate but will be correct when added to area['APb']
        area['APb'][idx] += area['Pan'][idx] # This isn't accurate but will be correct when added to area['AcP']
    
    # Voronoi area for each vertex of each face.
    # Again, recall the areas are always signed.
    vorf = np.stack((
        area['AcP'] + area['APb'],
        area['BaP'] + area['BPc'],
        area['CbP'] + area['CPa']), axis=1)
    
    # Accumulate vertex Voronoi areas for faces that meet at that vertex
    vorv = np.zeros(nvtx)
    np.add.at(vorv, fac, vorf)
    return vorv, vorf, areaABC

def calc_rot(n): # ============================================================
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

def calc_curvatures(vtx, fac, vtxn, facn, vorv, vorf): # ======================
    '''
    Compute curvatures k1 and k2 following the algorithm proposed by
    Rusinkiewicz (2004), as well as the corresponding directions.

    Parameters
    ----------
    vtx : NumPy array with 3 columns (float).
        Vextex coordinates.
    fac : NumPy array with 3 columns (int).
        Face indices.
    vtxn : NumPy array with 3 columns (float).
        Vertex normals.
    facn : NumPy array with 3 columns (float).
        Face normals.
    vorv : NumPy vector (float)
        Voronoi area per vertex.
    vorf : NumPy array with 3 columns (float).
        Voronoi area per vertex per face.

    Returns (as a dictionary)
    -------
    k1 : NumPy vector (float)
        Curvature k1.
    k2 : NumPy vector (float)
        Curvature k2.
    kd1 : NumPy array with 3 columns (float).
        Direction of curvature k1.
    kd2 : NumPy array with 3 columns (float).
        Direction of curvature k2.
    '''
    
    # Number of vertices and faces
    nvtx = vtx.shape[0]
    nfac = fac.shape[0]
    
    # Precompute transforms from the global to local coordinate system of each vertex
    rotv = np.zeros((3,3,nvtx))
    for v in range(nvtx):
       rotv[:,:,v] = calc_rot(vtxn[v])
    
    # Allocate space to store the Weingarten matrix for each vertex
    IIv = np.zeros((2,2,nvtx))
    
    # Allocate soace to store k1, k2, and the directions kd1 and kd2
    k1  = np.zeros(nvtx)
    k2  = np.zeros(nvtx)
    kd1 = np.zeros((nvtx,3))
    kd2 = np.zeros((nvtx,3))

    # For each face
    start_time = time.time()
    for f in range(nfac):
        if progress:
            progress_bar(f, nfac, start_time, prefix='Processing faces:', min_update_interval=1)
            
        # Transformation from the global the local coordinate system of this face
        rotf = calc_rot(facn[f])

        # Axes (uf,vf,wf) of the local face coordinate system
        uf = np.array([1,0,0])
        vf = np.array([0,1,0])
        
        # Confirm that the face normal matches the z of the coordinate system.
        # This must be (0,0,1); uncomment to test
        # print(facn[f] @ rotf)
        
        # Vertex coordinates in the local coordinate system, centered at the face barycenter
        vA = vtx[fac[f,0]] @ rotf
        vB = vtx[fac[f,1]] @ rotf
        vC = vtx[fac[f,2]] @ rotf
        
        # Normals in the local coordinate system
        nA = vtxn[fac[f,0]] @ rotf
        nB = vtxn[fac[f,1]] @ rotf
        nC = vtxn[fac[f,2]] @ rotf
        
        # Edges (coordinate of one vertex in relation to the other)
        # computed clockwise
        eA = vC - vB  # V_{i}
        eB = vA - vC  # V_{i+1} for vA
        eC = vB - vA  # V_{i+1} for vB when cycling back
        
        # Let's call our second fundamental tensor or Weingarten matrix as II.
        # II = [E F; F G], per Equation 1 of the Rusinkiewicz (2004) paper.
        # We have 3 unknowns, E, F, and G, which we can find via least squares.
        # Expanding the terms of the unnamed Figure/Equation between
        # Equations 5 and 6 of the Rusinkiewicz paper:
        # E*eA*u + F*eA*v = (nC-nB)*u
        # F*eA*u + G*eA*v = (nC-nB)*v
        # E*eB*u + F*eB*v = (nA-nC)*u
        # F*eB*u + G*eB*v = (nA-nC)*v
        # E*eC*u + F*eC*v = (nB-nA)*u
        # F*eC*u + G*eC*v = (nB-nA)*v
        # Reorganizing the terms for least squares:
        # eA*u*E + eA*v*F +    0*G = (nC-nB)*u
        #    0*E + eA*u*F + eA*v*G = (nC-nB)*v
        # eB*u*E + eB*v*F +    0*G = (nA-nC)*u
        #    0*E + eB*u*F + eB*v*G = (nA-nC)*v
        # eC*u*E + eC*v*F +    0*G = (nB-nA)*u
        #    0*E + eC*u*F + eC*v*G = (nB-nA)*v
        # So now we can do X*[E F G]' = y, where X has the coefficients (known),
        # [E F G]' is a column vector of unknowns, and Y has the normal
        # differences (also known).
        X = np.array([
                [eA@uf, eA@vf,     0],
                [    0, eA@uf, eA@vf],
                [eB@uf, eB@vf,     0],
                [    0, eB@uf, eB@vf],
                [eC@uf, eC@vf,     0],
                [    0, eC@uf, eC@vf] ])
        y = np.array([
                [(nC-nB)@uf],
                [(nC-nB)@vf],
                [(nA-nC)@uf],
                [(nA-nC)@vf],
                [(nB-nA)@uf],
                [(nB-nA)@vf] ])
        E,F,G = np.squeeze(np.linalg.lstsq(X, y, rcond=None)[0])
        IIf   = np.array([[E,F],[F,G]])
        
        # For each vertex of this face
        for v in range(3):
            
            # Reexpress IIf in terms of the vertex coordinate system.
            # First we define a matrix to "unrotate" the local coordinate system 
            # from the face back to the global and then to the vertex.            
            rotfv = rotf.T @ rotv[:,:,fac[f,v]]
            
            # Confirm that the vertex normal in the face local coordinate system
            # is successfully rotated so that it matches the z of the vertex coordinate system.
            # This must be (0,0,1); uncomment to test
            # print(vtxn[fac[f,v]] @ rotf @ rotfv)
            
            # Then we reexpress the tensor in the vertex local coordinate system
            # and weight it by the Voronoi area of this vertex at this face.
            IIv[:,:,fac[f,v]] += rotfv[0:2,0:2].T @ IIf @ rotfv[0:2,0:2] * vorf[f,v]
            
    # For each vertex of the mesh
    for v in range(nvtx):
            
        # Normalize the accummulated IIv by the total Voronoi area of this vertex
        IIv[:,:,v] /= vorv[v]
        
        # Compute principal curvatures and their directions (eigenvalues and eigenvectors)
        kvals, kdirs = np.linalg.eig(IIv[:,:,v])

        # Put back in the global coordinate system, from the vertex local coordinate system
        kdirs = np.vstack((kdirs, np.zeros((1,2)))).T
        kdirs = kdirs @ rotv[:,:,v].T
        
        # Annoyingly, and after many hours debugging, np.linalg.eig does not return
        # eigenvalues and eigenvectors in any particular order. Let's sort them...
        idx = np.argsort(kvals)[::-1] # sort in descending order
        
        # Store principal curvatures for subsequent saving
        k1[v]    = kvals[idx[0]]
        k2[v]    = kvals[idx[1]]
        kd1[v,:] = kdirs[idx[0],:]
        kd2[v,:] = kdirs[idx[1],:]
        
        # Confirm that the two principal directions are orthogonal to the normal
        # at this vertex. These two values must be zero; uncomment to test
        # print(np.dot(kd1[v,:],vtxn[v,:]), np.dot(kd2[v,:],vtxn[v,:]))
    return {'k1':k1, 'k2':k2, 'kdir1':kd1, 'kdir2':kd2}

def calc_composites(curvs): # ================================================
    
    # Gaussian curvature
    curvs['K']     = curvs['k1']*curvs['k2']

    # Mean curvature
    curvs['H']     = (curvs['k1']+curvs['k2'])/2

    # Gaussian-related: - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Gauss Curvature L^2 Norm (GLN)
    curvs['GLN']   = curvs['K']**2
    
    # Intrinsic Curvature Index (ICI, NICI, AICI)
    curvs['ICI']   = np.maximum(curvs['K'],0)
    curvs['NICI']  = np.minimum(curvs['K'],0) # Negative version
    curvs['AICI']  = np.absolute(curvs['K'])  # Absolute version
    
    # Area Fraction of Intrinsic Curvature Index
    curvs['FICI']  = (curvs['K'] > 0).astype(float)
    curvs['FNICI'] = (curvs['K'] < 0).astype(float)
    
    # SK2SK
    curvs['SK2SK'] = curvs['GLN'] / curvs['AICI']
    
    # Mean-related: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Mean Curvature L^2 Norm (MLN)
    curvs['MLN']   = curvs['H']**2
    
    # Mean Curvature Index (MCI, NMCI, AMCI)
    curvs['MCI']   = np.maximum(curvs['H'],0)
    curvs['NMCI']  = np.minimum(curvs['H'],0) # Negative version
    curvs['AMCI']  = np.absolute(curvs['H'])  # Absolute version

    # Area Fraction of Mean Curvature Index
    curvs['FMCI']  = (curvs['H'] > 0).astype(float)
    curvs['FNMCI'] = (curvs['H'] < 0).astype(float)
    
    # SH2SH
    curvs['SH2SH'] = curvs['MLN'] / curvs['AMCI']
    
    # Mixed:  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Curvature difference
    curvs['kdiff'] = curvs['k1'] - curvs['k2']
    
    # Folding Index
    curvs['FI']    = np.absolute(curvs['k1']) * (np.absolute(curvs['k1']) - np.absolute(curvs['k2']))
    
    # Curvedness Index
    curvs['CI']    = np.sqrt(curvs['k1']**2 + curvs['k2']**2)/np.sqrt(2)
    
    # Shape Index
    curvs['SI']    = 2*np.arctan((curvs['k1']+curvs['k2'])/(curvs['k1']-curvs['k2']))/np.pi

    return curvs

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(
                        description=\
                        'Compute various curvature indices. ' + \
                        'Invoke either with --subj (and optionally --subjdir and --srf) ' + \
                        'or with --in, --it, --out, and --ot. ' + \
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
                        help='Surface to compute curvatures from, if invoked with "--subj" (default: white)',
                        type=str, required=False, default='white')
    # Options to call with specific files
    parser.add_argument('--in', dest='filein',
                        help="Input surface file",
                        type=str, required=False, default=None)
    parser.add_argument('--it', dest='intype',
                        help="Input type (file format)",
                        type=str, required=False, default='surf')
    parser.add_argument('--outprefix', dest='outprefix',
                        help="Prefix for output curvature files",
                        type=str, required=False, default=None)
    parser.add_argument('--ot', dest='outtype',
                        help="Output type (file format)",
                        type=str, required=False, default='curv')
    # Show a nice progress bar (only use in interactive sessions)
    parser.add_argument('--progress',
                        help='Show a progress bar (avoid in headless systems)',
                        action='store_true', required=False, default=False)
    
    # Parsing proper
    args      = parser.parse_args()
    subj      = args.subj
    subjdir   = args.subjdir
    srf       = args.srf
    filein    = args.filein
    intype    = args.intype.lower()
    outprefix = args.outprefix
    outtype   = args.outtype.lower()
    progress  = args.progress
    
    # Check whether we are using a default FS subject structure, or isolated, custom files
    customargs = [filein, outprefix]
    if subj is None and None in customargs:
        raise SyntaxError('Either "--subj" or both "--in" and "--out" must be provided.')
    if subj is not None and not None in customargs:
        raise SyntaxError('Cannot use "--subj" together with "--in or "--outprefix".')
    
    # If a list of subjects was provided, let's use it
    if subj is not None:
        # Find SUBJECTS_DIR
        if subjdir is None:
            subjdir = os.getenv('SUBJECTS_DIR')
        if subjdir == '' or subjdir is None:
            raise SyntaxError('Either --subjdir option must be provided, or the environmental variable SUBJECTS_DIR must be set')
        # Make a list of subjects
        subjlist = subj.split(',')   
        H = ['lh', 'rh']
    else:
        subjlist = [None]
        H = [None]
    
    # For each subject and each hemisphere
    for subj in subjlist:
        for h in H:
            # Read input file
            if subj is None or h is None:
                print('Reading input file: {}'.format(filein))
                surfs = ('pial', 'white', 'inflated', 'sphere', 'sphere.reg', 'orig')
                if intype == 'obj' or filein.lower().endswith('.obj'):
                    obj = read_obj(filein)
                    vtx = obj['v']
                    fac = obj['f']
                elif intype == 'surf' or filein.lower().endswith(surfs):
                    vtx, fac, _, _ = read_surf(filein)
            else:
                filein = os.path.join(subjdir, subj, 'surf', '{}.{}'.format(h, srf))
                print('Reading input file: {}'.format(filein))
                vtx, fac, _, _ = read_surf(filein)
                
            # Compute vertex normals
            print('Computing face and vertex normals using the Max (1999) algorithm')
            vtxn, facn = calc_normals(vtx, fac)
            
            # Compute Voronoi areas
            print('Computing Voronoi areas per vertex at each face using an inhouse algorithm')
            vorv, vorf, areas = calc_voronoi_areas(vtx, fac)
        
            # Compute principal curvatures
            print('Computing principal curvatures using the Rusinkiewicz (2004) algorithm')
            curvs = calc_curvatures(vtx, fac, vtxn, facn, vorv, vorf)
            
            # Compute composite measures based on k1 and k2
            print('Computing composite curvatures from the principal curvatures')
            curvs = calc_composites(curvs)
            
            # Include Voronoi areas in the set, so it's easier to save
            curvs['varea'] = vorv
            
            # Save results
            if subj is None or h is None:
                print('Saving results with prefix: {}'.format(outprefix))
                if outtype in ['asc', 'dpv', 'curv']:
                    # FreeSurfer curvature format (ASCII or not)
                    for c in curvs:
                        if   c == 'k1':
                            dirs = curvs['kdir1']
                        elif c == 'k2':
                            dirs = curvs['kdir2']
                        elif c in ['kdir1', 'kdir2']:
                            continue
                        else:
                            dirs = None
                        fileout = '{}.{}.{}'.format(outprefix, c, outtype)
                        if args.outtype.lower() in ['asc', 'dpv']:
                            write_curv(fileout, curvs[c], vectors=dirs, use_ascii=True)
                        else:
                            write_curv(fileout, curvs[c], vectors=dirs, use_ascii=False)
                elif outtype in ['mgh', 'mgz']:
                    # FreeSurfer voxelwise format (mgh/mgz)
                    for c in curvs:
                        fileout = '{}.{}.{}'.format(outprefix, c, outtype)
                        if c in ['kdir1', 'kdir2']:
                            continue
                        write_mgh(fileout, np.asarray(curvs[c], dtype=np.float32).reshape(-1, 1, 1))
            else:
                outdir = os.path.join(subjdir, subj, 'after', 'curvs')
                print('Saving results to directory: {}'.format(outdir))
                os.makedirs(outdir, exist_ok=True)
                for c in curvs:
                    if   c == 'k1':
                        dirs = curvs['kdir1']
                    elif c == 'k2':
                        dirs = curvs['kdir2']
                    elif c in ['kdir1', 'kdir2']:
                        continue
                    else:
                        dirs = None
                    fileout = os.path.join(outdir, '{}.{}'.format(h, c))
                    write_curv(fileout, curvs[c], vec=dirs, use_ascii=False)
    print('Finished computing curvatures.')