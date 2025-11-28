#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 13:10:05 2025

@author: winkler
"""

import os
import sys
import argparse
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import shutil
import bids

import lib

# =====================================================================
# Helper Functions
# =====================================================================

def run_cmd(cmd: List[str],
            check: bool = True,
            env: Optional[dict] = None,
            capture_output: bool = True) -> subprocess.CompletedProcess:
    '''Run a command and print it.'''
    print(f'Running: {" ".join(cmd)}')
    return subprocess.run(cmd, check=check, text=True, capture_output=capture_output, env=env)

def check_fs_status(sub_dir: Path) -> Tuple[str, Optional[int]]:
    '''Check recon-all status from recon-all-status.log.'''
    log_file = sub_dir / 'scripts' / 'recon-all-status.log'
    if not log_file.exists():
        return 'failed', None

    with open(log_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return 'failed', None

    last_line = lines[-1]
    if 'exited with ERRORS' in last_line:
        return 'failed', None
    elif 'finished without error' in last_line:
        try:
            timestamp_str = last_line.split('at')[-1].strip()
            timestamp = datetime.strptime(timestamp_str, '%a %b %d %H:%M:%S %Z %Y')
            tdelta = int(datetime.now().timestamp()) - int(timestamp.timestamp())
            return 'success', tdelta
        except Exception:
            return 'unknown', None
    return 'unknown', None

def check_files(filelist: List[Path], step: str):
    '''Exit if required files are missing.'''
    missing = [f for f in filelist if not f.exists()]
    if missing:
        print(f'Error: FreeSurfer did not complete up to stage needed for {step}.')
        print('Missing files:')
        for f in missing:
            print(f'- {f}')
        sys.exit(1)

# =====================================================================
# Main Function
# =====================================================================

# Parse arguments
parser = argparse.ArgumentParser(
    description     = 'Run recon-all with additional steps.',
    formatter_class = argparse.RawTextHelpFormatter,
    epilog          = textwrap.dedent('''
    _________________________________________
    Anderson M. Winkler
    The University of Texas Rio Grande Valley
    Nov/2025
    https://brainder.org'''))
    
# Subject ID
parser.add_argument('--sub',
                    type     = str,
                    required = True,
                    help     = 'Subject ID in FreeSurfer (does not need to match the subject ID in BIDS (--bids_sub).')

# Non-BIDS inputs
parser.add_argument('--t1w',
                    nargs    = '+',
                    default  = None,
                    type     = Path,
                    help     = 'Input T1w files for this subject; more than one can be supplied.')
parser.add_argument('--t2w',
                    nargs    = '+',
                    type     = Path,
                    default  = None,
                    help     = 'Input T2w files for this subject; activates myelin proxy and pial refinement.')
parser.add_argument('--flair',
                    nargs    = '+',
                    type     = Path,
                    default  = None,
                    help     = 'Input FLAIR files for this subject; activates myelin proxy and pial refinement.')

# BIDS inputs
parser.add_argument('--bids_dir',
                    type     = Path,
                    default  = None,
                    help     = 'Input BIDS directory.')
parser.add_argument('--bids_db',
                    type     = Path,
                    default  = None,
                    help     = 'Path to directory of SQLite database that indices this BIDS directory.')
parser.add_argument('--bids_sub',
                    type     = str,
                    default  = None,
                    help     = 'Subject ID in the input BIDS directory (without the "sub-" prefix).')
parser.add_argument('--bids_filter',
                    type     = Path,
                    default  = None,
                    help     = 'Path to JSON filter to specify BIDS files to import.')
parser.add_argument('--bids_use_t2w',
                    action   = 'store_true',
                    default  = False,
                    help     = 'Use the T2w from BIDS to refine pial and compute a myelin proxy.')
parser.add_argument('--bids_use_flair',
                    action   = 'store_true',
                    default  = False,
                    help     = 'Use the FLAIR from BIDS to refine pial and compute a myelin proxy.')

# Where to store
parser.add_argument('--sd',
                    type     = Path,
                    default  = None,
                    help     = 'Subjects directory (overrides $SUBJECTS_DIR).')

# What to do
parser.add_argument('--all',
                    action   = 'store_true',
                    help     = 'Do everything (FWHM=10 unless --smooth used).')
parser.add_argument('--base',
                    action   = 'store_true',
                    help     = 'Process up to surface registration.')
parser.add_argument('--retessellate',
                    action   = 'store_true',
                    help     = 'Retessellate to fsaverage.')
parser.add_argument('--refine',
                    choices  = ['no', 'auto', 'T2', 'FLAIR'],
                    default  = 'no',
                    help     = 'Refine pial with T2/FLAIR.')
parser.add_argument('--curvatures',
                    action   = 'store_true',
                    help     = 'Compute curvature metrics.')
parser.add_argument('--lgi',
                    action   = 'store_true',
                    help     = 'Compute local gyrification index.')
parser.add_argument('--myelin',
                    action   = 'store_true',
                    help     = 'Compute myelin proxy from T2/FLAIR.')
parser.add_argument('--subregions', '--subseg',
                    dest     = 'subseg',
                    action   = 'store_true',
                    help     = 'Segment amygdala, thalamus, etc.')
parser.add_argument('--segpve', 
                    action   = 'store_true',
                    help     = 'Compute partial volume effects.')
parser.add_argument('--smooth',
                    nargs    = '+',
                    type     = float,
                    default  = None,
                    help     = 'Smooth with given FWHM.')
parser.add_argument('--views',
                    action   = 'store_true',
                    help     = 'Capture orthogonal views.')

args = parser.parse_args()

# -----------------------------------------------------------------
# Set up paths and flags
# -----------------------------------------------------------------
afterdir        = Path(__file__).resolve().parent
sub             = args.sub
bids_dir        = args.bids_dir.resolve()    if args.bids_dir    else None
bids_db         = args.bids_db.resolve()     if args.bids_db     else None
bids_filter     = args.bids_filter.resolve() if args.bids_filter else None
bids_use_t2w    = args.bids_use_t2w
bids_use_flair  = args.bids_use_flair
subjects_dir    = args.sd.resolve() if args.sd is not None else Path(os.environ.get('SUBJECTS_DIR'))
do_all          = args.all
do_retessellate = args.retessellate or do_all
do_refine       = args.refine
do_curvatures   = args.curvatures   or do_all
do_lgi          = args.lgi          or do_all
do_myelin       = args.myelin       or do_all
do_subseg       = args.subseg       or do_all
do_segpve       = args.segpve       or do_all
do_views        = args.views        or do_all
do_smooth       = args.smooth is not None or do_all
fwhm            = args.smooth if args.smooth is not None else [10.0]

# FreeSurfer and FSL directories (from the environment)
fs_home = os.environ.get('FREESURFER_HOME')
fsl_dir = os.environ.get('FSLDIR')

# -----------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------

if not (Path(fs_home) / 'bin' / 'recon-all').exists():
    raise ValueError('Error: FreeSurfer not configured.')

if not subjects_dir or not subjects_dir.is_dir():
    raise ValueError('Error: SUBJECTS_DIR not set or invalid.')

if do_all or do_myelin:
    if not (Path(fsl_dir) / 'bin' / 'fslmaths').exists():
        raise ValueError('Error: FSL not configured.')
        
if bids_dir is not None:
    if args.t1w is not None or args.t2w is not None or args.flair is not None:
        raise ValueError('Cannot use --bids_dir together with additional input files from --t1w, --t2w or --flair.')
    if args.bids_sub is None:
        raise ValueError('A BIDS subject (--bids_sub) must be specified (following the BIDS specification, not include the prefix "sub-" when specifying the BIDS subject ID).')

# -----------------------------------------------------------------
# Define input files
# -----------------------------------------------------------------

inputs = {'T1w':   [],
          'T2w':   [],
          'FLAIR': []}

if bids_dir is None:
    
    if args.t1w:
        inputs['T1w']   += [f.resolve() for f in args.t1w]
    if args.t2w:
        inputs['T2w']   += [f.resolve() for f in args.t2w]
    if args.flair:
        inputs['FLAIR'] += [f.resolve() for f in args.flair]

else:
    
    # Read the BIDS data (from dir or indexed)
    if bids_db is None:
        BIDS = bids.BIDSLayout(root = bids_dir)
    else:
        BIDS = bids.BIDSLayout(root = bids_dir,
                               database_path = bids_db)
        
    # Filter to select specific images
    if bids_filter is not None:
        bids_filter = lib.io.read_json(bids_filter)
    
    # Get T1w
    if bids_filter is None:
        query = {'subject': args.bids_sub, 'extension': ['.nii', '.nii.gz'], 'suffix': 'T1w'}
    else:
        query = {'subject': args.bids_sub, 'extension': ['.nii', '.nii.gz'], 'suffix': 'T1w', **bids_filter['T1w']}
    files = BIDS.get(**query)
    inputs['T1w'] += [f.path for f in files]
    
    # Get T2w
    if bids_use_t2w:
        if bids_filter is None:
            query = {'subject': args.bids_sub, 'extension': ['.nii', '.nii.gz'], 'suffix': 'T2w'}
        else:
            query = {'subject': args.bids_sub, 'extension': ['.nii', '.nii.gz'], 'suffix': 'T2w', **bids_filter['T2w']}
        files = BIDS.get(**query)
        inputs['T2w'] += [f.path for f in files]

    # Get FLAIR
    if bids_use_flair:
        if bids_filter is None:
            query = {'subject': args.bids_sub, 'extension': ['.nii', '.nii.gz'], 'suffix': 'FLAIR'}
        else:
            query = {'subject': args.bids_sub, 'extension': ['.nii', '.nii.gz'], 'suffix': 'FLAIR', **bids_filter['FLAIR']}
        files = BIDS.get(**query)
        inputs['FLAIR'] += [f.path for f in files]

# Print list of inputs
print('Input files:')
for key in inputs:
    print('- {}:'.format(key))
    if inputs[key]:
        for f in inputs[key]:
            print('  {}'.format(f))
    else:
        print('  None')

if not inputs['T1w']:
    raise ValueError('No input T1w files identified.')

# Directory of this subject
sub_dir = subjects_dir / sub

# =================================================================
# Base recon-all up to surfreg (first call to FreeSurfer)
# =================================================================
if inputs['T1w']:
    t1w_list = []
    for t1w in inputs['T1w']:
        t1w_list += ['-i']
        t1w_list += [str(t1w)]
    cmd = [ f'{fs_home}/bin/recon-all',
        '-s', sub, '-sd', str(subjects_dir),
        '-autorecon1', '-autorecon2', '-sphere', '-surfreg',
        '-norandomness' ] + t1w_list
    run_cmd(cmd)
    status, tdelta = check_fs_status(sub_dir)
    if status != 'success' or tdelta is None or tdelta > 10:
        sys.exit('Error: First call to recon-all failed or did not finish recently.')

# =================================================================
# Average multiple T2w and/or FLAIR
# =================================================================
(sub_dir / 'after').mkdir(exist_ok=True, parents=True)
for mod in ['T2w', 'FLAIR']:
    if len(inputs[mod]) > 1:
        cmd = [str(afterdir / 'stereomc')]
        for img in inputs[mod]:
            cmd = cmd + ['-i', img]
        cmd = cmd + ['-t', inputs[mod][0]]
        cmd = cmd + ['-o', str(sub_dir / 'after' / 'avg' / f'{mod}.nii.gz')]
        run_cmd(cmd)
        inputs[mod] = [sub_dir / 'after' / 'avg' / f'{mod}.nii.gz']

# =================================================================
# Euler number
# =================================================================
for hemi in ['lh', 'rh']:
    orig_nofix = sub_dir / 'surf' / f'{hemi}.orig.nofix'
    euler_out  = sub_dir / 'after' / f'{hemi}.euler.txt'
    cmd = [f'{fs_home}/bin/mris_euler_number', str(orig_nofix)]
    result = run_cmd(cmd)
    if result.stdout.strip():
        euler = result.stdout.strip().split()[-1]
        euler_out.write_text(euler)

# =================================================================
# Define whether refine the pial
# =================================================================
t2pial = flairpial = False
if do_refine == 'T2':
    t2pial = True
elif do_refine == 'FLAIR':
    flairpial = True
elif do_refine == 'auto':
    if inputs['T2w']   or (sub_dir / 'mri' / 'T2.mgz'   ).exists() or (sub_dir / 'surf' / 'lh.pial.T2'   ).exists():
        t2pial = True
    if inputs['FLAIR'] or (sub_dir / 'mri' / 'FLAIR.mgz').exists() or (sub_dir / 'surf' / 'lh.pial.FLAIR').exists():
        flairpial = True
if t2pial and flairpial:
    sys.exit('Error: Cannot refine with both T2 and FLAIR. You must choose one explicitly.')
if   t2pial    and inputs['T2w']:
    refineopts = ['-T2',    inputs['T2w'][0],   '-T2pial']
elif flairpial and inputs['FLAIR']:
    refineopts = ['-FLAIR', inputs['FLAIR'][0], '-FLAIRpial']
else:
    refineopts = []

# =================================================================
# Retessellation
# =================================================================
if do_retessellate:
    filelist = [
        sub_dir / 'mri'  / 'brain.finalsurfs.mgz',
        sub_dir / 'surf' / 'lh.orig',
        sub_dir / 'surf' / 'rh.orig',
        sub_dir / 'surf' / 'autodet.gw.stats.lh.dat',
        sub_dir / 'surf' / 'autodet.gw.stats.rh.dat',
        sub_dir / 'surf' / 'lh.sphere.reg',
        sub_dir / 'surf' / 'rh.sphere.reg' ]
    check_files(filelist, 'retessellation')
    
    cmd = [ str(afterdir / 'retessellate'),
        '--subj', sub, '--srf', 'orig',
        '--subjdir', str(subjects_dir) ]
    run_cmd(cmd)

    retess_lh = sub_dir / 'after' / 'retess' / 'lh.orig.retess'
    retess_rh = sub_dir / 'after' / 'retess' / 'rh.orig.retess'
    if not (retess_lh.exists() and retess_rh.exists()):
        sys.exit('Error: Retessellation failed. Missing output files.')
    shutil.copy(retess_lh, sub_dir / 'surf' / 'lh.orig')
    shutil.copy(retess_rh, sub_dir / 'surf' / 'rh.orig')

# =================================================================
# Second recon-all from white-preaparc until the end
# =================================================================
recon_steps = [
    '-white-preaparc', '-cortex-label', '-smooth2', '-inflate2',
    '-curvHK', '-sphere', '-surfreg', '-jacobian_white', '-avgcurv',
    '-cortparc', '-white', '-pial', '-curvstats', '-cortribbon',
    '-cortparc2', '-cortparc3', '-pctsurfcon', '-hyporelabel',
    '-aparc2aseg', '-apas2aseg', '-wmparc', '-parcstats',
    '-parcstats2', '-parcstats3', '-segstats', '-balabels' ]
cmd = [ f'{fs_home}/bin/recon-all',
    '-s', sub, '-sd', str(subjects_dir) ] + recon_steps + refineopts
run_cmd(cmd)
status, tdelta = check_fs_status(sub_dir)
if status != 'success' or tdelta is None :
    sys.exit('Error: Second call to recon-all failed.')

# Midthickness
for hemi in ['lh', 'rh']:
    mid = sub_dir / 'surf' / f'{hemi}.midthickness'
    if not mid.exists():
        cmd = [ f'{fs_home}/bin/mris_expand',
            '-thickness',
            str(sub_dir / 'surf' / f'{hemi}.white'),
            '0.5', str(mid) ]
        run_cmd(cmd)

# Distance orig to white.preaparc, indicating how much the retessellation caused movements
for hemi in ['lh', 'rh']:
    cmd = [ str(afterdir / 'surfdist'),
        '--ref', str(sub_dir / 'surf' / f'{hemi}.orig'),
        '--mov', str(sub_dir / 'surf' / f'{hemi}.white.preaparc'),
        '--out', str(sub_dir / 'after' / 'retess' / f'{hemi}.orig2whitepreaparc') ]
    run_cmd(cmd)

# =================================================================
# Curvatures
# =================================================================
if do_curvatures:
    filelist = [sub_dir / 'surf' / f'{h}.{s}' for h in ['lh', 'rh'] for s in ['white', 'pial']]
    check_files(filelist, 'curvature computation')
    cmd = [ str(afterdir / 'curvatures'),
        '--subj', sub, '--subjdir', str(subjects_dir),
        '--surf', 'pial,white' ]
    run_cmd(cmd)
    cmd = [str(afterdir / 'mantle'), '--subj', sub, '--subjdir', str(subjects_dir)]
    run_cmd(cmd)

# =================================================================
# Local Gyrification Index
# =================================================================
if do_lgi:
    filelist = [sub_dir / 'surf' / f'{h}.pial' for h in ['lh', 'rh']]
    check_files(filelist, 'LGI computation')
    env = os.environ.copy()
    env['PATH'] = f'{afterdir}:{env.get("PATH", "")}'
    cmd = [ f'{fs_home}/bin/recon-all',
        '-s', sub, '-sd', str(subjects_dir), '-localGI' ]
    print(env['PATH'])
    run_cmd(cmd, env=env, capture_output=False)

# =================================================================
# Myelin proxy
# =================================================================
if do_myelin:
    cmd = [str(afterdir / 'melina'), '-s', sub, '-c']
    run_cmd(cmd)

# =================================================================
# Subsegmentation
# =================================================================
if do_subseg:
    for structure in ['thalamus', 'hippo-amygdala', 'brainstem']:
        cmd = [ f'{fs_home}/bin/segment_subregions',
            '--cross', sub, '--sd', str(subjects_dir), structure ]
        run_cmd(cmd)
    for tool in ['mri_sclimbic_seg', 'mri_segment_hypothalamic_subunits']:
        cmd = [f'{fs_home}/bin/{tool}', '--s', sub, '--sd', str(subjects_dir)]
        run_cmd(cmd)

# =================================================================
# Partial Volume Effects
# =================================================================
if do_segpve:
    opts = '--myelin' if (do_myelin) else ''
    cmd = [ str(afterdir / 'segpve'),
        '--subj', sub, '--subjdir', str(subjects_dir), opts ]
    run_cmd(cmd)

# =================================================================
# Views
# =================================================================
if do_views:
    cmd = [ str(afterdir / 'views'),
        '--subj', sub, '--subjdir', str(subjects_dir), '--all' ]
    run_cmd(cmd)

# =================================================================
# Smoothing
# =================================================================
if do_smooth:
    for hemi in ['lh', 'rh']:
        mid = sub_dir / 'surf' / f'{hemi}.midthickness'
        if not mid.exists():
            cmd = [ f'{fs_home}/bin/mris_expand',
                '-thickness', str(sub_dir / 'surf' / f'{hemi}.white'),
                '0.5', str(mid) ]
            run_cmd(cmd)

    for f in fwhm:
        cmd = [ str(afterdir / 'smooth'),
            '--method', 'fs', '--subj', sub, '--fwhm', str(f) ]
        run_cmd(cmd)
    
        if do_views:
            cmd = [ str(afterdir / 'views'),
                '--subj', sub, '--subjdir', str(subjects_dir),
                '--all', '--fwhm', str(f) ]
            run_cmd(cmd)

print('recon-after completed successfully.')
