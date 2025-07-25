#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 13:43:37 2025

@author: winkler
"""

import os
import numpy as np
import nibabel as nib
import json

# =============================================================================
def read_surf(filein, is_ascii=False):
    '''
    Read a FreeSurfer geometry file.
    To put them in the same space as shown in FreeView, after loading,
    redefine vtx as vtx = vtx + info['cras']. FreeView does not seem to make
    use of info['xras'], info['yras'], and info['zras'],
    '''
    if is_ascii:
        with open(filein, 'r') as f:
            # Parse header and dimensions
            header = f.readline()
            if not header.startswith('#!ascii'):
                raise ValueError("Not a valid FreeSurfer ASCII file")            
            nV, nF = map(int, f.readline().split())
        
        # Read all data at once using numpy
        data = np.loadtxt(filein, skiprows=2)
        
        # Split into vertices and faces
        vtx   = data[:nV,:3]
        fac   = data[nF:,:3].astype(int)
        info  = None
        stamp = None
    else:
        vtx, fac, info, stamp = nib.freesurfer.read_geometry(filein, read_metadata=True, read_stamp=True)
    return vtx, fac, info, stamp

# -----------------------------------------------------------------------------
def write_surf(fileout, vtx, fac, info, stamp=True):
    '''
    Write a FreeSurfer geometry file.
    If you manually redefined the vtx as vtx = vtx + info['cras'], and
    you are supplying info, then need to subtract info['cras'] back.
    '''
    nib.freesurfer.write_geometry(fileout, vtx, fac, volume_info=info, create_stamp=True)
    
# =============================================================================
def read_mgh(filein):
    mgh    = nib.load(filein)
    data   = mgh.get_fdata()
    affine = mgh.affine
    header = mgh.header
    return data, affine, header

# -----------------------------------------------------------------------------
def write_mgh(fileout, data, affine=None, header=None):
    if affine is None:
        affine = np.eye(4)
    if header is None:    
        mgh = nib.MGHImage(data, affine)
    else:
        mgh = nib.MGHImage(data, affine, header)
    nib.save(mgh, fileout)

# =============================================================================
def read_annot(filein, orig_ids=False):
    labels, ctab, names = nib.freesurfer.io.read_annot(filein, orig_ids=orig_ids)
    names = [n.decode('utf-8') for n in names]
    return labels, ctab, names

# =============================================================================
def read_curv(filein, is_ascii=False):
    if is_ascii:
        data = np.loadtxt(filein, usecols=(1, 2, 3, 4)) # ignore col 0
        curv = data[:, 3]
        vec  = data[:,:3]
    else:
        curv = nib.freesurfer.io.read_morph_data(filein)
        vec  = None
    return curv, vec

# -----------------------------------------------------------------------------
def write_curv(fileout, curv, vec=None, use_ascii=False):
    '''
    Write a FreeSurfer curvature file.
    If "vec" is provided and not saving in ASCII, then an ASCII file (.dpv)
    will also be saved as it can store the vectors.
    
    Parameters
    ----------
    fileout : string
        Filename of the output.
    curv : NumPy vector
        Curvature (e.g., k1, k2 vertexise; can be any other quantity).
    vec : NumPy array with 3 columns.
        Direction of the curvature.
    use_ascii : bool
        Whether to save in ASCII format (aka .dpv, data-per-vertex)
        
    Returns
    -------
    None.
    '''
    if use_ascii:
        nvtx = curv.shape[0]
        idx  = np.arange(0,nvtx)[:,None]
        if vec is None:
            vec = np.zeros((nvtx,3))
        data = np.concatenate((idx, vec, curv[:,None]), axis=1)
        np.savetxt(fileout, data, fmt='%d %f %f %f %f')
    else:
        nib.freesurfer.io.write_morph_data(fileout, np.asarray(curv, dtype=np.float32))
        if vec is not None:
            fileout = fileout + '.dpv'
            write_curv(fileout, curv, vec=vec, use_ascii=True)
    return

# ============================================================================
def read_json(jsonfile):
    """Read a json file to dict."""
    with open(jsonfile, 'r') as fp:
        J = json.load(fp)
    return J

# -----------------------------------------------------------------------------
def write_json(jsonfile, J, epoch=None): 
    """Write a dict to json file."""
    with open(jsonfile, 'w') as fp:
        json.dump(J, fp, indent=2)
    if epoch is not None:
        os.utime(jsonfile, (epoch,epoch))
    return

def read_obj(filein): # =======================================================
    '''
    Open and parse a Wavefront OBJ file.
    Handles all standard face formats and returns texture/normal indices.
    
    Parameters:
    filein (str): Path to the OBJ file
    
    Returns:
    dict: Dictionary containing vertices, normals, texture coordinates, and faces
          with their respective vertex, texture, and normal indices
    '''
    v  = []
    vt = []
    vn = []
    f  = []
    ft = []
    fn = []
    with open(filein, 'r') as fp:
        for line in fp:
            if line.startswith('#'):  # Skip comments
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':     # Vertex coordinates
                v.append([float(x) for x in values[1:4]])
            elif values[0] == 'vt':  # Texture coordinates
                vt.append([float(x) for x in values[1:3]])
            elif values[0] == 'vn':  # Normals
                vn.append([float(x) for x in values[1:4]])
            elif values[0] == 'f':  # Face
                vidx  = []
                vtidx = []
                vnidx = []
                for val in values[1:]:
                    idx = val.split('/')
                    vidx.append(int(idx[0]) - 1)
                    if len(idx) > 1 and idx[1]: # Texture coordinate index
                        vtidx.append(int(idx[1]) - 1)
                    else:
                        vtidx.append(None)
                    if len(idx) > 2: # Normal coordinate index
                        vnidx.append(int(idx[2]) - 1)
                    else:
                        vnidx.append(None)
                f.append(vidx)
                ft.append(vtidx)
                fn.append(vnidx)
            elif values[0] == 's':
                continue
    obj = {
        'v':  np.array(v),   # Vertex coords
        'vt': np.array(vt),  # Vertex texture
        'vn': np.array(vn),  # Vertex normal
        'f':  np.array(f),   # Face indices (refer to v)
        'ft': np.array(ft),  # Face texture indices (refer to vt)
        'fn': np.array(fn)   # Face normal indices (refer to vn)
    }
    return obj
