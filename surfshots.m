function surfshots(varargin)
% Create a set of surface views in native geometry, useful for QC.
%
% surfshots('/path/to/my/project/subjects'
%
% Images will be placed in the directory "after", within each subject.
% 
% _____________________________________
% Anderson M. Winkler
% Univ. of Texas Rio Grande Valley
% May/2025
% http://brainder.org

% PALM must be available in the path. Edit these lines as needed.
palmdir = getenv('PALMDIR');
addpath(palmdir)
if ~ exist('palm_viewsurf','file')
    error('PALM must be available.')
end

% Get SUBJECTS_DIR
if nargin == 0
    sd = getenv('SUBJECTS_DIR');
else
    sd = varargin{1};
end
if isempty(sd)
    error('SUBJECTS_DIR cannot be empty');
end

% List all subjects
D = dir(sd);
for d = numel(D):-1:1
    if D(d).name(1) == '.'
        D(d) = [];
    end
end

% Iterate over each subject
S = {D(:).name};
for s = 1:numel(S)
    fprintf('Working on subject %s (%d/%d)\n',S{s},s,numel(S));
    tic;
    surfshotsf(fullfile(sd,S{s}));
    close all;
    toc
end
end

function surfshotsf(subjdir) % ============================================
% Operates on each subject, making surface captures and saving as PNG
% files. Customize as you wish

% Create output directory
outdir = fullfile(subjdir,'after');
if ~ exist(outdir, 'dir')
    mkdir(outdir);
end

% Surface shot 1
if ~ exist(fullfile(outdir,'bh.inflated.aparc.png'),'file')
    data = { ...
        fullfile(subjdir,'label','lh.aparc.annot'), ...
        fullfile(subjdir,'label','rh.aparc.annot') };
    surfs = { ...
        fullfile(subjdir,'surf','lh.inflated'), ...
        fullfile(subjdir,'surf','rh.inflated') };
    f = palm_viewsurf(data,surfs,'layout','strip','background',[0 0 0],'mapname','annot','inflated',true,'camlight',true,'facecolor','flat');
    set(f,'Position',[0 0 2400 300])
    print('-dpng',fullfile(outdir,'bh.inflated.aparc.png'));
end

% Surface shot 2
if ~ exist(fullfile(outdir,'bh.pial.aparc.png'),'file')
    data = { ...
        fullfile(subjdir,'label','lh.aparc.annot'), ...
        fullfile(subjdir,'label','rh.aparc.annot') };
    surfs = { ...
        fullfile(subjdir,'surf','lh.pial'), ...
        fullfile(subjdir,'surf','rh.pial') };
    f = palm_viewsurf(data,surfs,'layout','strip','background',[0 0 0],'mapname','annot','inflated',false,'camlight',true,'facecolor','flat');
    set(f,'Position',[0 0 2400 300])
    print('-dpng',fullfile(outdir,'bh.pial.aparc.png'));
end

% Surface shot 3
if ~ exist(fullfile(outdir,'bh.white.thickness.png'),'file')
    data = { ...
        fullfile(subjdir,'surf','lh.thickness'), ...
        fullfile(subjdir,'surf','rh.thickness') };
    surfs = { ...
        fullfile(subjdir,'surf','lh.white'), ...
        fullfile(subjdir,'surf','rh.white') };
    f = palm_viewsurf(data,surfs,'layout','strip','background',[0 0 0],'mapname','plasma','inflated',false,'camlight',false,'datarange',[0 5]);
    set(f,'Position',[0 0 2400 300])
    print('-dpng',fullfile(outdir,'bh.white.thickness.png'));
end

% Surface shot 4
if ~ exist(fullfile(outdir,'bh.white.curv.png'),'file')
    data = { ...
        fullfile(subjdir,'surf','lh.curv'), ...
        fullfile(subjdir,'surf','rh.curv') };
    surfs = { ...
        fullfile(subjdir,'surf','lh.white'), ...
        fullfile(subjdir,'surf','rh.white') };
    f = palm_viewsurf(data,surfs,'layout','strip','background',[.5 .5 .5],'mapname','coolhot4','inflated',false,'camlight',false,'datarange',[-.5 .5]);
    set(f,'Position',[0 0 2400 300])
    print('-dpng',fullfile(outdir,'bh.white.curv.png'));
end
end