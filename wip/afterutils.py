#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 13:43:37 2025

@author: winkler
"""

import nibabel as nib

def read_surf(file):
    vtx, fac, info, stamp = nib.freesurfer.read_geometry(file, read_metadata=True, read_stamp=True)
    return vtx, fac, info, stamp

def write_surf(file, vtx, fac, info, stamp=True):
    nib.freesurfer.write_geometry(file, vtx, fac, volume_info=info, create_stamp=True)