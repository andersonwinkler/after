#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 20:56:06 2025

@author: winkleram
"""
import os
import numpy as np
import nibabel as nib
import glob
import argparse
import pyvista as pv
import matplotlib.colors as mcolors
from PIL import Image
import gc

def load_surf(subjdir, sub, surf, specs): # ===================================
    '''
    Load surface for left and right hemis, and concatenate them too

    Parameters
    ----------
    subjdir : str
        Path to subjects' dir ($SUBJECTS_DIR).
    sub : str
        Subject (a directory within subjdir).
    surf : str
        Name of a FreeSurfer surface, e.g., white, pial, inflated, sphere, sphere.reg, etc.
        or a path to that surface (filename must start with lh or rh)

    Returns
    -------
    vtx : dict
        Vertex coordinates for lh, rh, and bh.
    fac : dict
        Face indices for lh, rh, and bh.
    '''
    
    # Filenames for lh and rh
    surfname = {}
    if os.path.exists(surf):
        fpth, fnam = os.path.split(surf)
        if fnam.startswith('lh'):
            surfname['lh'] = surf
            fnam[0:2] = 'rh'
            surfname['rh'] = os.path.join(fpth, fnam)
        elif fnam.startswith('rh'):
            surfname['rh'] = surf
            fnam[0:2] = 'lh'
            surfname['lh'] = os.path.join(fpth, fnam)
        else:
            raise ValueError('Surface file name must start with "lh" or "rh".')
    else:
        fnam = os.path.join(subjdir, sub, specs['surf'][surf]['dir'], 'lh.{}'.format(surf))
        if os.path.exists(fnam):
            surfname['lh'] = fnam
        else:
            raise ValueError('File does not exist: {}.'.format(fnam))
        fnam = os.path.join(subjdir, sub, specs['surf'][surf]['dir'], 'rh.{}'.format(surf))
        if os.path.exists(fnam):
            surfname['rh'] = fnam
        else:
            raise ValueError('File does not exist: {}.'.format(fnam))
    
    # Load surface
    vtx = {}
    fac = {}
    for h in ['lh', 'rh']:
        vtx[h], fac[h] = nib.freesurfer.read_geometry(surfname[h])
    
    # Shift coords a bit for inflated or sphere
    if 'inflated' in surf or 'sphere' in surf:
        vtx['lh'][:,0] = vtx['lh'][:,0] - np.max(vtx['lh'][:,0]) - 10
        vtx['rh'][:,0] = vtx['rh'][:,0] - np.min(vtx['rh'][:,0]) + 10
    vtx['bh'] = np.concatenate((vtx['lh'], vtx['rh']))
    
    # Create "both" hemispheres
    vtx['bh'] = np.concatenate((vtx['lh'], vtx['rh']))
    fac['bh'] = np.concatenate((fac['lh'], fac['rh'] + len(vtx['lh'])))
    for h in fac: # we need to add a column with "3" as needed by VTK/PyVista
        fac[h] = np.hstack((3*np.ones((fac[h].shape[0],1)).astype(int),fac[h]))
    return vtx, fac

def load_data(subjdir, sub, meas, specs): # ===================================
    '''
    Load overlay measure

    Parameters
    ----------
    subjdir : str
        Path to subjects' dir ($SUBJECTS_DIR).
    sub : str
        Subject (a directory within subjdir).
    meas : str
        Overlay measure, e.g., thichness, sulc, curv, etc.
    specs: dict
        A dictionary with a subdict under 'meas' from which some info is obtained.
        
    Returns
    -------
    dat : dict
        Data overlay for lh, rh, and bh.
    rgb : NumPy array
        Color map (for annot files)
    '''
    
    # Filenames for the measure
    measname = {}
    if os.path.exists(meas):
        fpth, fnam = os.path.split(meas)
        if fnam.startswith('lh'):
            measname['lh'] = meas
            fnam[0:2] = 'rh'
            measname['rh'] = os.path.join(fpth, fnam)
        elif fnam.startswith('rh'):
            measname['rh'] = meas
            fnam[0:2] = 'lh'
            measname['lh'] = os.path.join(fpth, fnam)
        else:
            raise ValueError('Measure file name must start with "lh" or "rh".')
    else:
        fnam = os.path.join(subjdir, sub, specs['meas'][meas]['dir'], 'lh.{}'.format(meas))
        if os.path.exists(fnam):
            measname['lh'] = fnam
        else:
            raise ValueError('File does not exist: {}.'.format(fnam))
        fnam = os.path.join(subjdir, sub, specs['meas'][meas]['dir'], 'rh.{}'.format(meas))
        if os.path.exists(fnam):
            measname['rh'] = fnam
        else:
            raise ValueError('File does not exist: {}.'.format(fnam))
    
    # Load data overlay
    dat = {}
    if meas.endswith('.mgz') or meas.endswith('.mgh'):
        for h in ['lh', 'rh']:
            dat[h] = np.squeeze(nib.load(measname[h]).get_fdata())
        rgb = None
    elif meas.endswith('annot'):
        for h in ['lh', 'rh']:
            lab, ctab, _ = nib.freesurfer.io.read_annot(measname[h], orig_ids=False)
            dat[h] = lab2fac(lab, fac[h])
        rgb = ctab[:,0:3]
    else:
        for h in ['lh', 'rh']:
            dat[h] = nib.freesurfer.io.read_morph_data(measname[h])
        rgb = None
    
    # Create "both" hemispheres
    dat['bh'] = np.concatenate((dat['lh'], dat['rh']))
    return dat, rgb


def lab2fac(lab, fac): # ======================================================
    '''
    Take vertexwise labels and transform into facewise labels
    by majority voting. If no consensus, the final vertex has precedence.

    Parameters
    ----------
    lab : NumPy array, nvtx by 1
        Labels (generally integers).
    fac : NumPy Array, nfac by 3
        Vertex indices that determine a face.

    Returns
    -------
    newlab : Numpy array, nfac by 1
        Labels for each face.

    '''
    newlab = np.zeros((fac.shape[0],1), dtype=int)
    lab[lab<0] = 0
    labf = lab[fac]
    idx0 = (labf[:,0] == labf[:,1]) | (labf[:,0] == labf[:,2])
    newlab[idx0,0] = labf[idx0,0] # vtx1 agrees with vtx2 or vtx3
    idx1 = labf[:,1] == labf[:,2]
    newlab[idx1,0] = labf[idx1,1] # vtx2 agrees with vtx3
    idx2 = ~(idx0 | idx1)
    newlab[idx2,0] = labf[idx2,2] # vtx3 has the final say
    return newlab

def plot_fig(vtx, fac, dat, rgb, outdir, mesh, meas, specs): # ================
    '''
    Plot figures. Variables are as in the other functions.
    '''
    # Define colormaps and color limits
    if meas.endswith('annot'):
        if meas not in specs['meas']:
            specs['meas'][meas] = {}
        specs['meas'][meas]['cmap'] = mcolors.LinearSegmentedColormap.from_list('', rgb/255)
        specs['meas'][meas]['clim'] = None
    elif meas not in specs['meas']:
        cmin = np.min(dat['bh'])
        cmax = np.max(dat['bh'])
        if cmin*cmax < 0:
            cmap = 'bwr'
            mx = max(abs(cmin),abs(cmax))
            clim = [-mx, mx]
        else:
            cmap = 'viridis'
            clim = [cmin, cmax]
        specs['meas'][meas] = {'cmap': cmap, 'clim': clim}

    # Meshes for subsequent plotting
    mesh = {}
    for h in ['lh','rh','bh']:
        mesh[h] = pv.PolyData(vtx[h], fac[h])
        mesh[h]['scalars'] = dat[h][:,None]
    
    # Plot each view
    for view in specs['views']:
        if   view in ['lhlat', 'lhmed']:
            h = 'lh'
        elif view in ['rhlat', 'rhmed']:
            h = 'rh'
        else:
            h = 'bh'
        
        # Render the image, save as a file  
        p = pv.Plotter(window_size=(2000, 2000), off_screen=True)
        p.background_color = (0, 0, 0)
        p.add_mesh(mesh[h], clim=specs['meas'][meas]['clim'], cmap=specs['meas'][meas]['cmap'], show_scalar_bar=False)
        p.view_xz()
        p.camera.elevation = specs['views'][view]['elevation']
        p.camera.azimuth   = specs['views'][view]['azimuth']
        p.screenshot(os.path.join(outdir,'{}_{}_{}.png'.format(surf, meas, view)))
        
        # Resize the image to create a thumbnail
        img = Image.open(os.path.join(outdir,'{}_{}_{}.png'.format(surf, meas, view)))
        img.thumbnail((200, 200), Image.LANCZOS)
        img.save(os.path.join(outdir,'thumbnails','{}_{}_{}.png'.format(surf, meas, view)))
        pv.close_all()
        del p
        gc.collect()
        
def makehtml(htmldir, subjlist, surf, meas, specs): # =========================
    '''
    Make HTML file. Variables are as in the other functions.
    '''
    os.makedirs(os.path.join(htmldir), exist_ok=True)
    with open(os.path.join(htmldir,'{}_{}.html'.format(surf,meas)), 'w', encoding='utf-8') as f:
        f.write('<html><body bgcolor="#222222"><table>\n')
        for subj in subjlist:
            shotsdir = os.path.relpath(os.path.join(subjdir, subj, 'after', 'shots'), start=htmldir)
            f.write('<tr>')
            for view in specs['views']:
                f.write('<td><a href="{}"><img src="{}" border=0 title="{}"></a></td>\n'.format(
                    os.path.join(shotsdir, '{}_{}_{}.png'.format(surf, meas, view)),
                    os.path.join(shotsdir, 'thumbnails','{}_{}_{}.png'.format(surf, meas, view)),
                    '{}, {}, {}, {}'.format(subj, surf, meas, view)))
            f.write('</tr>')
        f.write('</table></body></html>')

######## MAIN FUNCTION ########################################################
if __name__ == "__main__":
    
    # Argument parser
    parser = argparse.ArgumentParser(description='Plot surface views for each subject, and makes an HTML page for visualization.')
    parser.add_argument('--subj',    type=str, help='List of subjects separated by commas', required=False, default=None)
    parser.add_argument('--subjdir', type=str, help='Subjects directory (usually SUBJECTS_DIR)', required=False, default=None)
    parser.add_argument('--htmldir', type=str, help='HTML directory', required=False, default=None)
    args    = parser.parse_args()
    subjdir = args.subjdir
    if args.subjdir is None or args.subjdir == '':
        subjdir = os.getenv('SUBJECTS_DIR')
    if subjdir == '' or subjdir is None:
        raise ValueError('Either --subjdir option must be provided, or the environmental variable SUBJECTS_DIR must be set')

    # Make a list of subjects
    if args.subj is None or args.subj == '':
        tmp = glob.glob(subjdir + '/*')
        subjlist = []
        for t in tmp:
            tpth, tnam = os.path.split(t)
            if not tnam.startswith('fsaverage') and os.path.exists(os.path.join(t, 'scripts', 'recon-all.log')):
                subjlist.append(tnam)
    else:
        subjlist = args.subj.split(',')
        
    # Surfaces and measures to plot
    surflist = ['inflated','pial','white']
    measlist = ['aparc.annot', 'thickness', 'curv', 'sulc', 'jacobian_white', 'w-g.pct.mgh']
    
    # Define locations, colors, and view schemes
    specs = {}
    specs['surf'] = {
        'orig.nofix':      {'dir': 'surf'},
        'smoothwm.nofix':  {'dir': 'surf'},
        'inflated.nofix':  {'dir': 'surf'},
        'qsphere.nofix':   {'dir': 'surf'},
        'orig':            {'dir': 'surf'},
        'inflated':        {'dir': 'surf'},
        'white':           {'dir': 'surf'},
        'smoothwm':        {'dir': 'surf'},
        'sphere':          {'dir': 'surf'},
        'sphere.reg':      {'dir': 'surf'},
        'pial':            {'dir': 'surf'} }
    specs['meas'] = {
        'curv':                 {'dir': 'surf', 'cmap': 'bwr',    'clim': [-.5, .5]},
        'sulc':                 {'dir': 'surf', 'cmap': 'bwr',    'clim': [-12, 12]},
        'jacobian_white':       {'dir': 'surf', 'cmap': 'plasma', 'clim': [0, 2.5]},
        'thickness':            {'dir': 'surf', 'cmap': 'plasma', 'clim': [1, 5]},
        'w-g.pct.mgh':          {'dir': 'surf', 'cmap': 'plasma', 'clim': [0, 40]},
        'aparc.annot':          {'dir': 'label', 'cmap': None,    'clim': None},
        'aparc.a2009s.annot':   {'dir': 'label', 'cmap': None,    'clim': None},
        'aparc.DKTatlas.annot': {'dir': 'label', 'cmap': None,    'clim': None} }
    specs['views'] = {
        'lhlat': {'elevation':   0, 'azimuth': -90},
        'lhmed': {'elevation':   0, 'azimuth':  90},
        'rhlat': {'elevation':   0, 'azimuth':  90},
        'rhmed': {'elevation':   0, 'azimuth': -90},
        'bhsup': {'elevation':  90, 'azimuth':   0},
        'bhinf': {'elevation': -90, 'azimuth': 180},
        'bhant': {'elevation':   0, 'azimuth': 180},
        'bhpos': {'elevation':   0, 'azimuth':   0} }

    # For each subject, make all figures
    for subj in subjlist:
        
        # Where to save
        outdir = os.path.join(subjdir, subj, 'after', 'shots')
        os.makedirs(os.path.join(outdir,'thumbnails'), exist_ok=True)
        
        # For each surface and measure
        for surf in surflist:
            for meas in measlist:
                if os.path.exists(os.path.join(outdir,'thumbnails','{}_{}_{}.png'.format(surf, meas, 'bhpos'))):
                    print('Skipping: {}, {}, {}'.format(subj, surf, meas))
                else:
                    print('Working on: {}, {}, {}'.format(subj, surf, meas))
                    
                    # Load the data
                    vtx, fac = load_surf(subjdir, subj, surf, specs)
                    dat, rgb = load_data(subjdir, subj, meas, specs)
                   
                    # Plot and save main fig and thumbnail
                    plot_fig(vtx, fac, dat, rgb, outdir, surf, meas, specs)
    
    # For each combination of surfaces and meshes, make an HTML file
    if args.htmldir is not None and args.htmldir != '':
        for surf in surflist:
            for meas in measlist:
                makehtml(args.htmldir, subjlist, surf, meas, specs)
