#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat Jun  7 18:27:11 2025

@author: winkler
'''

import numpy as np
import math

from . import utils

# =============================================================================
def tetrahedron(meas='e', value=1):
    '''Generate tetrahedron vertices and faces.'''
    vtx = np.array([
        [1, 1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, -1, 1] ])
    vtx = vtx / (2 * np.sqrt(2)) # make edge = 1
    fac = np.array([
        [0, 2, 1],
        [1, 2, 3],
        [0, 3, 2],
        [0, 1, 3] ])
    scaling = {
            'e': value,
            'af': np.sqrt(4 * value / np.sqrt(3)),
            'at': np.sqrt(value / np.sqrt(3)),
            'v': (12 * value / np.sqrt(2)) ** (1/3),
            'cr': value * np.sqrt(8/3),
            'ir': value * np.sqrt(24) }
    vtx = vtx * scaling[meas]
    return vtx, fac

# -----------------------------------------------------------------------------
def hexahedron(meas='e', value=1):
    '''Generate hexahedron (cube) vertices and faces.'''
    vtx = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1] ])
    vtx = vtx / 2 # make edge = 1
    fac = np.array([
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [4, 0, 3, 7] ])
    scaling = {
            'e': value,
            'af': np.sqrt(value),
            'at': np.sqrt(value / 6),
            'v': value ** (1/3),
            'cr': 2 * value / np.sqrt(3),
            'ir': 2 * value }
    vtx = vtx * scaling[meas]
    return vtx, fac

# -----------------------------------------------------------------------------
def octahedron(meas='e', value=1):
    '''Generate octahedron vertices and faces.'''
    vtx = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [-1, 0, 0], [0, -1, 0], [0, 0, -1] ])
    vtx = vtx / np.sqrt(2) # make edge = 1
    fac = np.array([
        [0, 1, 2], [1, 3, 2], [3, 4, 2], [4, 0, 2],
        [1, 0, 5], [3, 1, 5], [4, 3, 5], [0, 4, 5] ])
    scaling = {
            'e': value,
            'af': np.sqrt(4 * value / np.sqrt(3)),
            'at': np.sqrt(value / (2 * np.sqrt(3))),
            'v': (3 * value / np.sqrt(2)) ** (1/3),
            'cr': 2 * value / np.sqrt(2),
            'ir': 6 * value / np.sqrt(6) }
    vtx = vtx * scaling[meas]
    return vtx, fac

# -----------------------------------------------------------------------------
def dodecahedron(meas='e', value=1):
    '''Generate dodecahedron vertices and faces.'''
    g = (1 + np.sqrt(5)) / 2  # Golden ratio
    vtx = np.array([
        [1/g, 0, g], [-1/g, 0, g], [1/g, 0, -g], [-1/g, 0, -g],
        [0, g, -1/g], [0, g, 1/g], [0, -g, -1/g], [0, -g, 1/g],
        [g, 1/g, 0], [g, -1/g, 0], [-g, 1/g, 0], [-g, -1/g, 0],
        [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
        [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1] ])
    vtx = vtx * g / 2 # make edge = 1
    fac = np.array([
        [12, 5, 13, 1, 0], [18, 7, 14, 0, 1], [16, 6, 19, 3, 2],
        [17, 4, 15, 2, 3], [4, 5, 12, 8, 15], [5, 4, 17, 10, 13],
        [6, 7, 18, 11, 19], [7, 6, 16, 9, 14], [12, 0, 14, 9, 8],
        [16, 2, 15, 8, 9], [17, 3, 19, 11, 10], [18, 1, 13, 10, 11] ])
    scaling = {
            'e': value,
            'af': np.sqrt(4 * value / np.sqrt(25 + 10 * np.sqrt(5))),
            'at': np.sqrt(value / (3 * np.sqrt(25 + 10 * np.sqrt(5)))),
            'v': (4 * value / (15 + 7 * np.sqrt(5))) ** (1/3),
            'cr': 4 * value / (np.sqrt(15) + np.sqrt(3)),
            'ir': 20 * value / np.sqrt(250 + 110 * np.sqrt(5)) }
    vtx = vtx * scaling[meas]
    return vtx, fac

# -----------------------------------------------------------------------------
def icosahedron(meas='e', value=1, fsmode=True):
    '''
    Generate icosahedron vertices and faces.
    If fsmode=True, will generate the coordinates as used in fsaverage,
    albeit with higher precision.
    '''
    if fsmode:
        sqrt5 = np.sqrt(5)
        vtx = np.array([
            [0, 0, 1],
            [ +(5-sqrt5)/10, -np.sqrt((5+sqrt5)/10), +1/sqrt5 ],
            [ 2/sqrt5, 0, 1/sqrt5 ],
            [ +(5-sqrt5)/10, +np.sqrt((5+sqrt5)/10), +1/sqrt5 ],
            [ -(5+sqrt5)/10, +np.sqrt((5-sqrt5)/10), +1/sqrt5 ],
            [ -(5+sqrt5)/10, -np.sqrt((5-sqrt5)/10), +1/sqrt5 ],
            [ -(5-sqrt5)/10, -np.sqrt((5+sqrt5)/10), -1/sqrt5 ],
            [ +(5+sqrt5)/10, -np.sqrt((5-sqrt5)/10), -1/sqrt5 ],
            [ +(5+sqrt5)/10, +np.sqrt((5-sqrt5)/10), -1/sqrt5 ],
            [ -(5-sqrt5)/10, +np.sqrt((5+sqrt5)/10), -1/sqrt5 ],
            [ -2/sqrt5, 0, -1/sqrt5 ],
            [0, 0, -1] ])
        vtx = vtx * np.sqrt(10+2*sqrt5)/4; # make edge = 1
        fac = np.array([
            [0, 3, 4], [ 0, 4, 5], [ 0, 5, 1], [ 0, 1, 2],
            [0, 2, 3], [ 3, 2, 8], [ 3, 8, 9], [ 3, 9, 4],
            [4, 9, 10], [ 4, 10, 5], [ 5, 10, 6], [ 5, 6, 1],
            [1, 6, 7], [ 1, 7, 2], [ 2, 7, 8], [ 8, 11, 9],
            [9, 11, 10], [10, 11, 6], [ 6, 11, 7], [ 7, 11, 8] ])
    else:
        g = (1 + np.sqrt(5)) / 2  # Golden ratio
        vtx = np.array([
            [0, 1, g], [0, -1, g], [0, 1, -g], [0, -1, -g],
            [1, g, 0], [-1, g, 0], [1, -g, 0], [-1, -g, 0],
            [g, 0, 1], [g, 0, -1], [-g, 0, 1], [-g, 0, -1] ])
        vtx = vtx / 2; # make edge = 1
        fac = np.array([
            [0, 1, 8], [0, 10, 1], [2, 9, 3], [2, 3, 11],
            [1, 7, 6], [3, 6, 7], [5, 0, 4], [2, 5, 4],
            [6, 9, 8], [8, 9, 4], [7, 10, 11], [5, 11, 10],
            [1, 6, 8], [0, 8, 4], [10, 7, 1], [10, 0, 5],
            [6, 3, 9], [7, 11, 3], [2, 4, 9], [2, 11, 5] ])
    scaling = {
            'e': value,
            'af': np.sqrt(4 * value / np.sqrt(3)),
            'at': np.sqrt(value / (5 * np.sqrt(3))),
            'v': (12 * value / (5 * (3 + np.sqrt(5)))) ** (1/3),
            'cr': 4 * value / np.sqrt(10 + 2 * np.sqrt(5)),
            'ir': 12 * value / np.sqrt(42 + 18 * np.sqrt(5)) }
    vtx = vtx * scaling[meas]
    return vtx, fac

# =============================================================================
def icotest(X):
    # Constants for the icosahedron
    V0 = 12
    F0 = 20
    E0 = 30

    # Discover what kind of data this is
    nX = len(X)
    orderv = math.log2((nX-2)/(V0-2))/2
    orderf = math.log2(nX/F0)/2
    ordere = math.log2(nX/E0)/2
    if orderv == np.round(orderv):
        dtype = 'vtx'
        order = orderv
    elif orderf == np.round(orderf):
        dtype = 'fac'
        order = orderf
    elif ordere == np.round(ordere):
        dtype = 'edg'
        order = ordere
    else:
        dtype = None
        order = -1
    return dtype, order

# =============================================================================
def icoup(vtx, fac, n, fsmode=True):
    '''
    Recursively upsample an icosahedron n times (will work also with tetra and octahedron).
    If fsmode=True, the ordering of vertices and faces will match that of FreeSurfer.
    n is the number of recursions, starting from the input vtx and fac
    (not the final order of the geodesic sphere).
    
    Parameters
    ----------
    vtx : NumPy array (num vertices by 3)
        Vertex coordinates (x,y,z).
    fac : NumPy array (num faces by 3)
        Face indices (all faces are triangular).
    n : int
        Number of times it will be iteratively upsampled.
    fsmode : bool
        Whether to order vertices and faces in the same scheme used by FreeSurfer
        in fsaverage. Not the most convenient but good for compatibility.
        
    Returns
    -------
    vtxnew : NumPy array (num vertices by 3)
        Vertex coordinates (x,y,z) of the upsampled surface.
    facnew : NumPy array (num faces by 3)
        Face indices of the upsampled surface.    
    '''

    # Get the radius (it should be the same for all vertices; take the mean just in case)
    r = np.mean(np.linalg.norm(vtx, axis=1))

    # Iterate over recursions (n times)
    for _ in range(n):
    
        # Number of vertices, edges, and faces
        nV1 = len(vtx)
        nF1 = len(fac)
        nE1 = nV1 + nF1 - 2
        nF2 = 4*nF1
    
        # Vertices for each face
        tri = vtx[fac]

        # Fork to produce FreeSurfer-compatible spheres, or
        # use a more convenient method
        if fsmode:
            
            # Edge midpoints
            midvtx = np.zeros((nE1*2,3))
            newfac = np.zeros((nF2,3)).astype(int)
            e1 = 0
            f2 = 0
            for f1 in range(nF1):
                # Mid-edge vertices (new vertices, contain duplicates)
                midvtx[e1+0] = (tri[f1,2] + tri[f1,0]) / 2
                midvtx[e1+1] = (tri[f1,1] + tri[f1,2]) / 2
                midvtx[e1+2] = (tri[f1,0] + tri[f1,1]) / 2
                # New face indices, also with duplicates
                newfac[f1]       = [fac[f1,0], e1+2+nV1,  e1+0+nV1]
                newfac[f2+0+nF1] = [e1+0+nV1,  e1+1+nV1,  fac[f1,2]]
                newfac[f2+1+nF1] = [e1+2+nV1,  e1+1+nV1,  e1+0+nV1]
                newfac[f2+2+nF1] = [e1+2+nV1,  fac[f1,1], e1+1+nV1]
                e1 += 3
                f2 += 3
            
            # Drop duplicated vertices
            mixvtx = np.vstack([vtx, midvtx])
            _, uidx, uinv = np.unique(mixvtx, axis=0, return_index=True, return_inverse=True)
            sidx   = np.sort(uidx)
            vtx    = mixvtx[sidx]
            
            # Renumber duplicated faces
            aidx   = np.argsort(np.argsort(uidx))
            sinv   = aidx[uinv]
            fac    = sinv[newfac]

        else:
            # Edge midpoints
            mid1 = (tri[:,0] + tri[:,1]) / 2
            mid2 = (tri[:,1] + tri[:,2]) / 2
            mid3 = (tri[:,2] + tri[:,0]) / 2
            midvtx = np.vstack([mid1, mid2, mid3])
            
            # Drop duplicates, assemble new vertices
            _, uidx = np.unique(np.round(midvtx, 10), axis=0, return_index=True)
            newvtx = midvtx[uidx]
            vtx = np.vstack([vtx, newvtx])
            
            # New faces
            newfac = []
            for i, f in enumerate(fac):
                v0, v1, v2 = f
                # Find indices of midpoints
                m01 = np.argmin(np.sum((newvtx-mid1[i])**2, axis=1)) + nV1
                m12 = np.argmin(np.sum((newvtx-mid2[i])**2, axis=1)) + nV1
                m20 = np.argmin(np.sum((newvtx-mid3[i])**2, axis=1)) + nV1
                # Create 4 new triangular faces
                newfac.extend([
                    [v0, m01, m20],
                    [m01, v1, m12],
                    [m20, m12, v2],
                    [m01, m12, m20] ])
            fac = np.array(newfac).astype(int)
        
        # Scale to unit norm then by the radius
        vtx = vtx / np.linalg.norm(vtx, axis=1)[:,None] * r

    return vtx, fac

# =============================================================================
def icodown(vtx, fac, n):
    '''
    Downsample a surface from from a higher-order tessellated icosahedron to
    a lower order one.

    Parameters
    ----------
    vtx : NumPy array (num vertices by 3)
        Vertex coordinates (x,y,z).
    fac : NumPy array (num faces by 3)
        Face indices (all faces are triangular).
    n : int
        Number of times it will be iteratively downsampled.

    Returns
    -------
    vtxnew : NumPy array (num vertices by 3)
        Vertex coordinates (x,y,z) of the downsampled surface.
    facnew : NumPy array (num faces by 3)
        Face indices of the downsampled surface.
    '''
    
    # Constants for the icosahedron
    V0 = 12
    F0 = 20
    
    # Iterate over recursions (n times)
    for _ in range(n):
    
        # Current icosahedron order
        nV    = vtx.shape[0]
        nF    = fac.shape[0]
        order = round(np.log((nV-2)/(V0-2))/np.log(4))
        
        # Remove vertices (keep only those needed for target order)
        nVnew = 4**(order-1) * (V0 - 2) + 2
        vtx   = vtx[:nVnew,:]
        
        # Remove face indices
        nFnew = 4**(order-1) * F0
        remap = np.arange(nV)
        for f in range(nF):
            v1, v2, v3 = fac[f,:]
            remap[v1] = min(remap[v1],v2,v3)
        facnew = np.zeros((nFnew,3), dtype=int)
        for f in range(nFnew):
            for v in range(3):
                facnew[f,v] = remap[fac[f,v]]
        fac = facnew
    return vtx, fac

# =============================================================================
def dpxdown(dpx, n, vtx=None, fac=None, fsmode=True, pycno=False):
    '''
    Downsample scalar field over a surface. Can be vertexwise or facewise.

    Parameters
    ----------
    dpx : NumPy vector
        Data to be downsampled (vertexwise or facewise)
    n : int
        Number of iterations
    vtx : NumPy array, num vertices by 3, optional
        Vertex coordinates, needed if downsampling non-FreeSurfer facewise data.
        The default is None.
    fac : NumPy array, num faces by 3, optional
        Face indices, needed if downsampling non-FreeSurfer facewise data.
        The default is None.
    fsmode : bool, optional
        Indicate if we are working with FreeSurfer ordering of vertices and indices.
        Since the ordering doesn't need to be inferred, it works much faster.
        The default is True.
    pycno : bool, optional
        For vertexwise data, use a mass-conservative (pycnophylactic) method.
        The default is False.

    Returns
    -------
    dpxnew : NumPy vector
        Downsampled data.
    '''
    # Constants for the icosahedron
    V0 = 12
    F0 = 20
    E0 = 30
    
    # Discover what kind of data this is
    nX = len(dpx)
    if fac is None:
        orderv = math.log2((nX-2)/(V0-2))/2
        orderf = math.log2(nX/F0)/2
        ordere = math.log2(nX/E0)/2
        if orderv == np.round(orderv):
            dtype = 'dpv'
            order = orderv
        elif orderf == np.round(orderf):
            dtype = 'dpf'
            order = orderf
        elif ordere == np.round(ordere):
            dtype = 'dpe'
            order = ordere
        else:
            raise ValueError('Number of datapoints does not match an icosahedron.')
        nV = 4**order*(V0-2)+2
        nF = 4**order*F0
        nE = 4**order*E0
        if vtx is not None and nV != vtx.shape[0]:
            raise ValueError('Vertex coordinate array does not match the size of the data')
    else:
        nV = int(np.max(fac))+1
        nF = len(fac)
        nE = nV + nF - 2
        if  nX == nV:
            dtype = 'dpv'
            order = math.log2((nX-2)/(V0-2))/2
        elif nX == nF:
            dtype = 'dpf'
            order = math.log2(nX/F0)/2
        elif nX == nE:
            dtype = 'dpe'
            order = math.log2(nX/E0)/2
        else:
            raise ValueError('Number of datapoints does not match the face indices provided.')
        if vtx is not None and nV != vtx.shape[0]:
            raise ValueError('Vertex coordinate array does not match the size of the data')
        
    if dtype == 'dpv':
        for _ in range(n):
            nVnew  = 4**(order-1)*(V0-2)+2
            dpxnew = dpx[0:nVnew]
            if pycno:
                dpxadd = np.zeros(dpxnew.shape)
                facs   = np.sort(fac, axis=1)
                idx    = facs[:,0] < nVnew
                np.add.at(dpxadd, facs[idx,0], dpx[facs[idx,1]])
                np.add.at(dpxadd, facs[idx,0], dpx[facs[idx,2]])
                dpx    = dpxnew + dpxadd/4
            order -= 1
                
    elif dtype == 'dpf':
        for _ in range(n):
            nF    = len(dpx)
            nFnew = int(4**(order-1)*F0)
            if fsmode: # if we know it's FreeSurfer, it's much faster
                dpxnew = dpx[0:nFnew]
                dpx = dpxnew + np.sum(dpx[nFnew:].reshape((nFnew,3), order='C'), axis=1)
            else:
                import sparse
                if hasattr(fac, 'dtype') and fac.dtype.byteorder == '>':
                    fac = fac.astype(np.int32)
                
                # Lookup tables showing:
                # - for each face given the vertex indices, the face index
                # - for each vertex pair to be deleted, the vertex that stays
                facs, idx = utils.sortrows(np.sort(fac, axis=1))
                lutf  = sparse.COO(facs.T, np.arange(nF)[idx], (nF,nF,nF))
                lutv  = sparse.COO(facs[:nFnew*3,1:].T, facs[:nFnew*3,0], (nFnew*3,nFnew*3))
                
                # Downsample
                vtxd, facd = icodown(vtx, fac, 1)
                
                # Lookup table showing the face number given the vertex indices
                facds = np.sort(facd, axis=1)
                lutd  = sparse.COO(facds.T, np.arange(nFnew), (nFnew, nFnew, nFnew))
                
                # Populate the new dpx
                dpxnew = np.zeros(nFnew)
                for f in range(nFnew*3, nF, 1):
                    v1 = lutv[facs[f,0],facs[f,1]]
                    v2 = lutv[facs[f,1],facs[f,2]]
                    v3 = lutv[facs[f,0],facs[f,2]]
                    sv1, sv2, sv3 = sorted([v1, v2, v3])
                    ft = lutd[sv1, sv2, sv3] # target face
                    dpxnew[ft] += dpx[lutf[ facs[f,0], facs[f,1], facs[f,2] ]]
                    dpxnew[ft] += dpx[lutf[ sv1,       facs[f,0], facs[f,1] ]]
                    dpxnew[ft] += dpx[lutf[ sv2,       facs[f,1], facs[f,2] ]]
                    dpxnew[ft] += dpx[lutf[ sv3,       facs[f,0], facs[f,2] ]]
                vtx = vtxd
                fac = facd
                dpx = dpxnew
            order -= 1
            
    elif dtype == 'dpe':
        raise ValueError('Downsampling for edgewise data has not yet been implemented')
    return dpx

# =============================================================================
def dpf2dpv(dpf, fac, facu=None, pycno=False, fsmode=True):
    '''
    Convert facewise to vertexwise.

    Parameters
    ----------
    dpf : NumPy vector
        Data per face to be converted to vertexwise.
    fac : NumPy array (num faces by 3)
        Vertex indices that form each face.
    facu : NumPy array (num faces "up" by 3), optional
        Similar to fac but with 4x more faces, for the icosahedron one
        order upward. Needed if we want to sample to vertexwise at
        that upward resolution
    pycno : Bool, optional
        Whether to use a mass-conservative (pycnophylactic) method.
        The default is False.
    fsmode : Bool, optional
        Assume face indices follow FreeSurfer structure. Only has an effect
        if facu is provided. The default is True.

    Returns
    -------
    dpv : NumPy vector
        Data resampled to vertexwise.
    '''
    
    # Number of vertices
    nV  = np.max(fac) + 1
    
    # If no upsampling is required
    if facu is None:
        facflat = fac.flatten()
        dpv = np.zeros(nV)
        np.add.at(dpv, facflat, np.tile(dpf[:,None],(1,3)).flatten())
        if pycno:
            # Redistribute by a factor of 1/3
            dpv /= 3
            # Scale factor to account for some vertices receiving data from
            # different number of faces (5 or 6) (not needed unless we assume
            # all faces have identical size, impossible in practice)
            # s   = np.sum(cnt)/nV/cnt
            # dpv = dpv*s
        else:
            # Average by the number of faces that meet at that vertex
            cnt = np.zeros(nV)
            np.add.at(cnt, facflat, 1)
            dpv /= cnt
    else:
        # Find additional vertex indices (of the edge midpoints) that the data
        # from this face needs to be interpolated to
        nVu = np.max(facu) + 1
        nF  = fac.shape[0]
        if fsmode:
            idx  = np.arange(facu.shape[0])
            idx = idx[nF:].reshape((nF,3), order='C')[:,1]
            faca = facu[idx]
        else:
            # Neighbors of the lowres vertices in the highres
            facus = utils.sortrows(np.sort(facu, axis=1))[0]
            neig = {}
            for v in range(nV):
                idx = facus[:,0] == v
                neig[v] = facus[idx,1:].flatten()
            # Find the edge midpoints
            faca = np.zeros(fac.shape).astype(int)
            for f in range(nF):
                v1, v2, v3 = fac[f]
                faca[f,0] = np.intersect1d(neig[v1], neig[v2])[0]
                faca[f,1] = np.intersect1d(neig[v2], neig[v3])[0]
                faca[f,2] = np.intersect1d(neig[v1], neig[v3])[0]
        # Now that we know these midpoint vertex indices, we can interpolate
        facflat  = fac.flatten()
        facaflat = faca.flatten()
        dpv      = np.zeros(nVu)
        cnt      = np.zeros(nVu)
        np.add.at(dpv, facflat,  np.tile(dpf[:,None],(1,3)).flatten())
        np.add.at(dpv, facaflat, np.tile(dpf[:,None],(1,3)).flatten())
        np.add.at(cnt, facflat,  1)
        np.add.at(cnt, facaflat, 1)
        if pycno:
            # Redistribute by a factor of 1/6 (since we have 6 vertices)
            dpv /= 6
            # Scale factor to account for some vertices receiving data from
            # different number of faces (5 or 6)
            s   = np.sum(cnt)/nVu/cnt
            dpv = dpv*s
        else:
            # Average by the number of faces that meet at that vertex
            dpv /= cnt
    return dpv

def make_fine_fsaverage(outdir=None):
    '''
    Make fsaverage surfaces (icosahedrons recursively subdivided) and save them
    in the module's tree structure. These are "finer" than the original shipped
    with FreeSurfer in that vertex coordinates have more decimal places, and
    recursions span the range 0-9, as opposed to 3-7.
    Instead of lh and rh (which are identical), saves as "xh".
    '''
    import os
    from . import io
    
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
    vtx, fac = icosahedron(meas='cr', value=100, fsmode=True)
    io.write_surf(os.path.join(outdir, 'xh.fsaverage0.fine.sphere.reg'), vtx, fac, info)
    
    # Recursively subdivide and save
    for n in range(1,10):
        print(f'Working on subdivision {n}')
        vtx, fac = icoup(vtx, fac, 1, fsmode=True)
        io.write_surf(os.path.join(outdir, f'xh.fsaverage{n}.fine.sphere.reg'), vtx, fac, info)