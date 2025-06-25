#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 18:23:07 2025

@author: winkler
"""

import numpy as np
import time

from . import platonic
from . import surf
from . import utils

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

# =============================================================================
def calc_normals(vtx, fac):
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

# =============================================================================
def signed_area(tri, rfn=None):
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
        Can be created as: tri = vtx[fac]
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

# =============================================================================
def calc_voronoi_areas(vtx, fac):
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

# =============================================================================
def calc_curvatures(vtx, fac, vtxn, facn, vorv, vorf, progress=False):
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
       rotv[:,:,v] = normal2zaxis(vtxn[v])
    
    # Allocate space to store the Weingarten matrix for each vertex
    IIv = np.zeros((2,2,nvtx))
    
    # Allocate soace to store k1, k2, and the directions kd1 and kd2
    k1  = np.zeros(nvtx)
    k2  = np.zeros(nvtx)
    kd1 = np.zeros((nvtx,3))
    kd2 = np.zeros((nvtx,3))

    start_time = time.time()
    for f in range(nfac):
        if progress:
            utils.progress_bar(f, nfac, start_time, prefix='Processing faces:', min_update_interval=1)
            
        # Transformation from the global the local coordinate system of this face
        rotf = normal2zaxis(facn[f])

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
        
        # Output as a dict, that can be expanded with other metrics
        curvs = {'k1':k1, 'k2':k2, 'kdir1':kd1, 'kdir2':kd2}
    return curvs

# =============================================================================
def calc_composites(curvs):
    
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

# =============================================================================
def avg_edge_len_per_face(tri):
    '''
    Compute the average edge length of each face.
    The input can be created as tri = vtx[fac]
    '''
    edges  = tri[:,[2,0,1],:] - tri[:,[1,2,0],:]
    avglen = np.sqrt(np.sum(edges**2, axis=2))
    avglen = np.mean(avglen, axis=1)
    return avglen

# =============================================================================
def calc_fd(vtx, fac, fsmode=True):
    '''
    Compute the fractal dimension using our inhouse method.

    Parameters
    ----------
    vtx : NumPy array num vertices by 3
        Vertex coordinates.
    fac : NumPy array num faces by 3
        DESCRIPTION.
    fsmode : bool, optional
        Boolean indicating if this surface if from FreeSurfer or not.
        Makes the computations much faster. The default is True.

    Returns
    -------
    D : Numpy vector, with num faces elements
        Fractal dimension, per face.
    '''
    
    # Area per face and avg edge length in the original resolution
    tri1   = vtx[fac]
    area1  = signed_area(tri1)[0]
    e1     = avg_edge_len_per_face(tri1)
  
    # Downsample these scalars (measured in original resolution)
    area1d = platonic.dpxdown(area1, 1, fsmode=True)
    e1d    = platonic.dpxdown(e1, 1, fsmode=True)/4
  
    # Measure ares in terms of edge lengths
    N1d    = area1d/e1d/e1d
  
    # Downsample surface
    vtx0, fac0 = platonic.icodown(vtx, fac, 1)
    
    # Area per face and avg edge length of downsampled surface
    tri0   = vtx0[fac0]
    area0  = signed_area(tri0)[0]
    e0     = avg_edge_len_per_face(tri0)
    
    # Measure area in terms of edge lengths
    N0     = area0/e0/e0
    
    # Fractal dimension, per downsampled face
    D = -np.log(N1d/N0)/np.log(e1d/e0)
    return D

# =============================================================================
def retessellate(vtx1, fac1, vtx2, fac2, vtx3, fac3, progress=False):
    '''
    Retessellate a mesh.

    Parameters
    ----------
    vtx1 : NumPy array (num vertices by 3)
        Vertex coordinates of the source sphere (typically ?h.sphere.reg).
    fac1 : NumPy arrau (num faces by 3)
        Face indicesindices of the source sphere (typically ?h.sphere.reg).
    vtx2 : NumPy array (num vertices by 3
        Vertex coordinates of the target sphere (typically ?h.sphere from fsaverage).
    fac2 : NumPy arrau (num faces by 3)
        Face indices of the target sphere (typically ?h.sphere from fsaverage).
    vtx3 : NumPy array (num vertices by 3
        Vertex coordinates of the mesh to be retessellated (e.g., ?h.white).
    fac3 : NumPy arrau (num faces by 3)
        Face indices of the mesh to be retessellated (e.g., ?h.white)
    progress : bool, optional
        Show a progress bar? The default is False.

    Returns
    -------
    vtx4 : NumPy array (num vertices by 3)
        Vertex coordinates of the retessellated mesh. Its face indices are fac2.
    '''

    # Default margin
    marg =  0.05;
    
    nF1 = fac1.shape[0]
    nV2 = vtx2.shape[0]

    # Where the result is going to be stored
    vtx4 = np.zeros((nV2, 3))
    
    # Vertices' coords per face
    facvtx1 = np.hstack((vtx1[fac1[:,0],:], vtx1[fac1[:,1],:], vtx1[fac1[:,2],:]))
    
    # Face barycenter
    xbary = np.mean(facvtx1[:, [0, 3, 6], None], axis=1)    # x-coordinate
    ybary = np.mean(facvtx1[:, [1, 4, 7], None], axis=1)    # y-coordinate
    zbary = np.mean(facvtx1[:, [2, 5, 8], None], axis=1)    # z-coordinate
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
            utils.progress_bar(f, nF1, start_time, prefix='Processing faces:', min_update_interval=1)
        
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
    return vtx4

# =============================================================================
def surf_dist(vtx1, vtx2, ord=2):
    '''
    Compute the distance between surfaces 1 and 2, specified by their
    matching vertex coordinates.
    (In fact, we compute the norm; ord can be any of those accepted by
     np.linalg.norm)

    Parameters
    ----------
    vtx1 : NumPy array, num vertices by 3
        Vertex coordinates of first surface.
    vtx2 : NumPy array, num vertices by 3
        Vertex coordinates of second surface.
        Must be the same number of vertices as the first.
    ord : {int, float, inf, -inf, ‘fro’, ‘nuc’}
        Order of the norm (see np.linalg.norm).
        Default is 2 (Euclidean distance)

    Returns
    -------
    dist : NumPy vector, num vertices
        Vertexwise norm (Euclidean distance by default).
    vec : NumPy array, num vertices by 3
        Vector over which the norm is calculated (vtx1 - vtx2).
    '''
    vec  = vtx1 - vtx2
    dist = np.linalg.norm(vec, ord=ord)
    return dist, vec

# =============================================================================
def calc_rpw(vtxp, vtxw, fac, relative=False):
    '''
    Compute the ratio pial/white areas, as defined by Mann et al (2021).
    However here we use Voronoi areas per vertex, as opposed to the typical
    method that assigns to each vertex 1/3 of the area of each triangle that
    meet at that vertex. The latter is also provided as a second output.
    
    Reference:
    * Mann C, Schäfer T, Bletsch A, Gudbrandsen M, Daly E, Suckling J,
      Bullmore ET, Lombardo MV, Lai M, Craig MC, MRC AIMS Consortium,
      Baron‐Cohen S, Murphy DGM, Ecker C. Examining volumetric gradients
      based on the frustum surface ratio in the brain in autism spectrum
      disorder. Hum Brain Mapp. 2021 Mar;42(4):953–966.

    Parameters
    ----------
    vtxp : NumPy array, num vertices by 3
        Vertex coordinates of pial surface.
    vtxw : NumPy array, num vertices by 3
        Vertex coordinates of white surface.
    fac : NumPy array, num faces by 3
        Vertex indices that define the faces, same for pial and white.
    relative: bool, optional
        Returns not the ratio r=p/w, but instead the ratio (p-w)/(p+w).

    Returns
    -------
    vorrpw : NumPy vector, num vertices
        Ratio surface area of pial by surface area of white, using
        the Voronoi areas.
    rpw : NumPy vector, num vertices
        Ratio surface area of pial by surface area of white, using
        the typical method (1/3 of face area given to each vertex).
    '''
    # Compute Voronoi areas and ratio, vertexwise
    vorvp, _, areafp  = surf.calc_voronoi_areas(vtxp, fac)
    vorvw, _, areafw  = surf.calc_voronoi_areas(vtxw, fac)
    
    # Convert area per face to typical area per vertex
    areavp = platonic.dpf2dpv(areafp, fac, facu=None, pycno=True)
    areavw = platonic.dpf2dpv(areafw, fac, facu=None, pycno=True)
    
    # Compute the ratios
    if relative: # Difference in relation to the mean
        vorrpw = (vorvp-vorvw)/(vorvp+vorvw)
        rpw    = (areavp-areavw)/(areavp+areavw)
    else: # Simple ratio, as in the original paper
        vorrpw = vorvp/vorvw
        rpw    = areavp/areavw
    return vorrpw, rpw

# =============================================================================
def calc_volume(vtxp, vtxw, fac, method='analytical'):
    '''
    Compute the volume of the cortical mantle, vertexwise.

    Parameters
    ----------
    vtxp : NumPy array, num vertices by 3
        Vertex coordinates of pial surface.
    vtxw : NumPy array, num vertices by 3
        Vertex coordinates of white surface.
    fac : NumPy array, num faces by 3
        Vertex indices that define the faces, same for pial and white.
    method : string, optional
        Name of the method. Can be 'analytical', 'product', or 
        some experimental options (check source code).
        The default is 'analytical'.

    Returns
    -------
    volv : NumPy vector, num vertices
        Volume, vertexwise.
    '''
    # Face coords for both surfaces
    trip = vtxp[fac]
    triw = vtxw[fac]    

    if method == 'analytical':

        # Vertex coordinates (ABC, for pial and white).
        # Use Ap as the origin (0,0,0)
        Ap = trip[:,0,:]
        Bp = trip[:,1,:] - Ap
        Cp = trip[:,2,:] - Ap
        Aw = triw[:,0,:] - Ap
        Bw = triw[:,1,:] - Ap
        Cw = triw[:,2,:] - Ap
        
        # Each obliquely truncated trilateral pyramid can be split into
        # three tetrahedra:
        # - T1: (Aw,Bw,Cw,Ap)
        # - T2: (Ap,Bp,Cp,Bw)
        # - T3: (Ap,Cp,Cw,Bw)
        # Since Ap is the common vertex for all three, it can be used as the origin.
        # The next lines compute the volume for each, using a scalar triple product:
        T1 = np.abs((Aw * np.cross(Bw, Cw, axis=1)).sum(axis=1))
        T2 = np.abs((Bp * np.cross(Cp, Bw, axis=1)).sum(axis=1))
        T3 = np.abs((Cp * np.cross(Cw, Bw, axis=1)).sum(axis=1))
    
        # Add them up (volume per face)
        volf = (T1 + T2 + T3) / 6
        
        # Convert to vertexwise
        volv = platonic.dpf2dpv(volf, fac, facu=None, pycno=True)
        
    elif method == 'product':
        
        # Facewise white area, converted to vertexwise
        areafw = signed_area(triw, rfn=None)[0]
        areavw = platonic.dpf2dpv(areafw, fac, facu=None, pycno=True)
        
        # Cortical thickness, calculated (as opposed to loaded from FS)
        ct   = surf_dist(vtxp, vtxw)
        
        # Volume, vertexwise
        volv = ct*(areavw)
        
    elif method == 'mann': # should give the same as 'analytical'
        
        # Facewise areas, converted to vertexwise
        areafp = signed_area(trip, rfn=None)[0]
        areafw = signed_area(triw, rfn=None)[0]
        areavp = platonic.dpf2dpv(areafp, fac, facu=None, pycno=True)
        areavw = platonic.dpf2dpv(areafw, fac, facu=None, pycno=True)
        
        # Cortical thickness, calculated (as opposed to loaded from FS)
        ct   = surf_dist(vtxp, vtxw)
        
        # Volume, vertexwise
        volv = ct*(areavp + areavw + np.sqrt(areavp*areavw))
    return volv