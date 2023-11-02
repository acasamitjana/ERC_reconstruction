% This is my own version of MRIread from Freesurfer.
% I made the following modifications:
% 1. It doesn't crash if MRI parameters (TE/TR) are not available, just
% throws a warning instead.
% 2. It is parfor-friendly.
% 3. It makes sure that file names are distinct in the temporary directory
% 4. Allows to specify the temporary directory
% Juan Eugenio Iglesias
% myMRIread(filename,justheader, tempdir)
function MRI = myMRIread(filename,justheader,tempdir)

if(exist('justheader','var')~=1) || isempty(justheader), justheader = 0; end
if(exist('tempdir','var')~=1) || isempty(tempdir), tempdir = '/tmp/'; end
if tempdir(end)~='/', tempdir=[tempdir '/']; end
if exist(tempdir,'dir')==0
    error('Error in myMRIread: temporary directory does not exist')
end

if length(filename)>6 && strcmp(filename(end-6:end),'.nii.gz')>0
    cl=clock();
    p1=num2str(round(1e6*cl(end)));
    [~,aux]=fileparts(filename);
    aux(aux=='.')='_';
    p2=aux;
    p3=getenv('PBS_JOBID');
    cl=clock();
    rng(round(1e6*cl(end))+sum(double(filename)));
    p4=num2str(round(1000000*rand(1)));
    fname2 = [tempdir p1 '_' p2 '_' p3 '_' p4 '.nii'];
    while exist(fname2,'file')>0
        pause(0.1*rand(1));
        cl=clock();
        p1=num2str(round(1e6*cl(end)));
        cl=clock();
        rng(round(1e6*cl(end))+sum(double(filename)));
        p4=num2str(round(1000000*rand(1)));
        fname2 = [tempdir p1 '_' p2 '_' p3 '_' p4 '.nii'];
    end
    system(['gunzip -c ' filename ' > ' fname2]);
    MRI=myMRIread_aux(fname2,justheader);
    delete(fname2);
elseif length(filename)>3 && strcmp(filename(end-3:end),'.mgz')>0
    cl=clock();
    p1=num2str(round(1e6*cl(end)));
    [~,aux]=fileparts(filename);
    aux(aux=='.')='_';
    p2=aux;
    p3=getenv('PBS_JOBID');
    cl=clock();
    rng('default')
    rng(round(1e6*cl(end))+sum(double(filename)));
    p4=num2str(round(1000000*rand(1)));
    fname2 = [tempdir p1 '_' p2 '_' p3 '_' p4 '.mgh'];
    while exist(fname2,'file')>0
        pause(0.1*rand(1));
        cl=clock();
        p1=num2str(round(1e6*cl(end)));
        cl=clock();
        rng(round(1e6*cl(end))+sum(double(filename)));
        p4=num2str(round(1000000*rand(1)));
        fname2 = [tempdir p1 '_' p2 '_' p3 '_' p4 '.mgh'];
    end
    system(['gunzip -c ' filename ' > ' fname2]);
    MRI=myMRIread_aux(fname2,justheader);
    delete(fname2);
else
    MRI=myMRIread_aux(filename,justheader);
end





function mri = myMRIread_aux(fstring,justheader)


mri = [];

[fspec fstem fmt] = MRIfspec(fstring);
if(isempty(fspec))
    err = sprintf('ERROR: cannot determine format of %s (%s)\n',fstring,mfilename);
    error(err);
    return;
end

mri.srcbext = '';    % empty be default
mri.analyzehdr = []; % empty be default
mri.bhdr = []; % empty be default

%-------------- MGH ------------------------%
switch(fmt)
    case {'mgh'}
        [mri.vol, M, mr_parms, volsz] = load_mgh(fspec,[],[],justheader);
        if(isempty(M))
            fprintf('ERROR: loading %s as MGH\n',fspec);
            mri = [];
            return;
        end
        if(~justheader)
            mri.vol = permute(mri.vol,[2 1 3 4]);
            volsz = size(mri.vol);
        else
            mri.vol = [];
            volsz(1:2) = [volsz(2) volsz(1)];
        end
        try
            tr = mr_parms(1);
            flip_angle = mr_parms(2);
            te = mr_parms(3);
            ti = mr_parms(4);
        catch
            tr=nan; flip_angle=nan; te=nan; ti=nan;
        end
        %--------------- bshort/bfloat -----------------------%
    case {'bhdr'}
        if(~justheader)
            [mri.vol bmri] = fast_ldbslice(fstem);
            if(isempty(mri.vol))
                fprintf('ERROR: loading %s as bvolume\n',fspec);
                mri = [];
                return;
            end
            volsz = size(mri.vol);
        else
            mri.vol = [];
            bmri = fast_ldbhdr(fstem);
            if(isempty(bmri))
                fprintf('ERROR: loading %s as bvolume\n',fspec);
                mri = [];
                return;
            end
            [nslices nrows ncols ntp] = fmri_bvoldim(fstem);
            volsz = [nrows ncols nslices ntp];
        end
        [nrows ncols ntp fs ns endian bext] = fmri_bfiledim(fstem);
        mri.srcbext = bext;
        M = bmri.T;
        tr = bmri.tr;
        flip_angle = bmri.flip_angle;
        te = bmri.te;
        ti = bmri.ti;
        mri.bhdr = bmri;
        %------- analyze -------------------------------------
    case {'img'}
        hdr = load_analyze(fspec,justheader);
        if(isempty(hdr))
            fprintf('ERROR: loading %s as analyze\n',fspec);
            mri = [];
            return;
        end
        volsz = hdr.dime.dim(2:end);
        indnz = find(volsz~=0);
        volsz = volsz(indnz);
        volsz = volsz(:)'; % just make sure it's a row vect
        if(~justheader) mri.vol = permute(hdr.vol,[2 1 3 4]);
        else            mri.vol = [];
        end
        volsz([1 2]) = volsz([2 1]); % Make consistent. No effect when rows=cols
        tr = 1000*hdr.dime.pixdim(5); % msec
        flip_angle = 0;
        te = 0;
        ti = 0;
        hdr.vol = []; % already have it above, so clear it
        M = vox2ras_1to0(hdr.vox2ras);
        mri.analyzehdr = hdr;
        %------- nifti nii -------------------------------------
    case {'nii'}
        hdr = load_nifti(fspec,justheader);
        if(isempty(hdr))
            fprintf('ERROR: loading %s as analyze\n',fspec);
            mri = [];
            return;
        end
        volsz = hdr.dim(2:end);
        indnz = find(volsz~=0);
        volsz = volsz(indnz);
        volsz = volsz(:)'; % just make sure it's a row vect
        if(~justheader) mri.vol = permute(hdr.vol,[2 1 3 4 5]);
        else            mri.vol = [];
        end
        volsz([1 2]) = volsz([2 1]); % Make consistent. No effect when rows=cols
        tr = hdr.pixdim(5); % already msec
        flip_angle = 0;
        te = 0;
        ti = 0;
        hdr.vol = []; % already have it above, so clear it
        M = hdr.vox2ras;
        mri.niftihdr = hdr;
        %---------------------------------------------------
    otherwise
        fprintf('ERROR: format %s not supported\n',fmt);
        mri = [];
        return;
end
%--------------------------------------%

mri.fspec = fspec;
mri.pwd = pwd;

mri.flip_angle = flip_angle;
mri.tr  = tr;
mri.te  = te;
mri.ti  = ti;

% Assumes indices are 0-based. See vox2ras1 below for 1-based.  Note:
% MRIwrite() derives all geometry information (ie, direction cosines,
% voxel resolution, and P0 from vox2ras0. If you change other geometry
% elements of the structure, it will not be reflected in the output
% volume. Also note that vox2ras still maps Col-Row-Slice and not
% Row-Col-Slice.  Make sure to take this into account when indexing
% into matlab volumes (which are RCS).
mri.vox2ras0 = M;

% Dimensions not redundant when using header only
volsz(length(volsz)+1:4) = 1; % Make sure all dims are represented
mri.volsize = volsz(1:3); % only spatial components
mri.height  = volsz(1);   % Note that height (rows) is the first dimension
mri.width   = volsz(2);   % Note that width (cols) is the second dimension
mri.depth   = volsz(3);
mri.nframes = volsz(4);

%--------------------------------------------------------------------%
% Everything below is redundant in that they can be derivied from
% stuff above, but they are in the MRI struct defined in mri.h, so I
% thought I would add them here for completeness.  Note: MRIwrite()
% derives all geometry information (ie, direction cosines, voxel
% resolution, and P0 from vox2ras0. If you change other geometry
% elements below, it will not be reflected in the output volume.

mri.vox2ras = mri.vox2ras0;

mri.nvoxels = mri.height * mri.width * mri.depth; % number of spatial voxles
mri.xsize = sqrt(sum(M(:,1).^2)); % Col
mri.ysize = sqrt(sum(M(:,2).^2)); % Row
mri.zsize = sqrt(sum(M(:,3).^2)); % Slice

mri.x_r = M(1,1)/mri.xsize; % Col
mri.x_a = M(2,1)/mri.xsize;
mri.x_s = M(3,1)/mri.xsize;

mri.y_r = M(1,2)/mri.ysize; % Row
mri.y_a = M(2,2)/mri.ysize;
mri.y_s = M(3,2)/mri.ysize;

mri.z_r = M(1,3)/mri.zsize; % Slice
mri.z_a = M(2,3)/mri.zsize;
mri.z_s = M(3,3)/mri.zsize;

ic = [(mri.width)/2 (mri.height)/2 (mri.depth)/2 1]';
c = M*ic;
mri.c_r = c(1);
mri.c_a = c(2);
mri.c_s = c(3);
%--------------------------------------------------%

%-------- The stuff here is for convenience --------------

% 1-based vox2ras. Good for doing transforms in matlab
mri.vox2ras1 = vox2ras_0to1(M);

% Matrix of direction cosines
mri.Mdc = [M(1:3,1)/mri.xsize M(1:3,2)/mri.ysize M(1:3,3)/mri.zsize];

% Vector of voxel resolutions (Row-Col-Slice)
mri.volres = [mri.xsize mri.ysize mri.zsize];

% Have to swap rows and columns back
voldim = [mri.volsize(2) mri.volsize(1) mri.volsize(3)]; %[ncols nrows nslices]
volres = [mri.volres(2)  mri.volres(1)  mri.volres(3)];  %[dcol drow dslice]
mri.tkrvox2ras = vox2ras_tkreg(voldim,volres);


return;




function hdr = load_nifti(niftifile,hdronly)
% hdr = load_nifti(niftifile,hdronly)
%
% Loads nifti header and volume. The volume is stored
% in hdr.vol. Columns and rows are not swapped.
%
% Handles compressed nifti (nii.gz) by issuing a unix command to
% uncompress the file to a temporary file, which is then deleted.
%
% Dimensions are in mm and msec
% hdr.pixdim(1) = physical size of first dim (eg, 3.125 mm or 2000 ms)
% hdr.pixdim(2) = ...
%
% The sform and qform matrices are stored in hdr.sform and hdr.qform.
%
% hdr.vox2ras is the vox2ras matrix based on sform (if valid), then
% qform.
%
% Handles data structures with more than 32k cols by looking for
% hdr.dim(2) = -1 in which case ncols = hdr.glmin. This is FreeSurfer
% specific, for handling surfaces. When the total number of spatial
% voxels equals 163842, then the volume is reshaped to
% 163842x1x1xnframes. This is for handling the 7th order icosahedron
% used by FS group analysis.
%
% See also: load_nifti_hdr.m
%


%
% load_nifti.m
%
% Original Author: Doug Greve
% CVS Revision Info:
%    $Author: nicks $
%    $Date: 2011/03/02 00:04:12 $
%    $Revision: 1.18 $
%
% Copyright © 2011 The General Hospital Corporation (Boston, MA) "MGH"
%
% Terms and conditions for use, reproduction, distribution and contribution
% are found in the 'FreeSurfer Software License Agreement' contained
% in the file 'LICENSE' found in the FreeSurfer distribution, and here:
%
% https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
%
% Reporting: freesurfer@nmr.mgh.harvard.edu
%

hdr = [];

if(nargin < 1 | nargin > 2)
    fprintf('hdr = load_nifti(niftifile,<hdronly>)\n');
    return;
end

if(~exist('hdronly','var')) hdronly = []; end
if(isempty(hdronly)) hdronly = 0; end

hdr = load_nifti_hdr(niftifile);
if(isempty(hdr))
    return;
end

% Check for ico7
nspatial = prod(hdr.dim(2:4));
IsIco7 = 0;
if(nspatial == 163842) IsIco7 = 1; end

% If only header is desired, return now
if(hdronly)
    if(IsIco7)
        % Reshape
        hdr.dim(2) = 163842;
        hdr.dim(3) = 1;
        hdr.dim(4) = 1;
    end
    return;
end

% Get total number of voxels
dim = hdr.dim(2:end);
ind0 = find(dim==0);
dim(ind0) = 1;
nvoxels = prod(dim);

% Open to read the pixel data
fp = fopen(niftifile,'r',hdr.endian);

% Get past the header
fseek(fp,round(hdr.vox_offset),'bof');

switch(hdr.datatype)
    % Note: 'char' seems to work upto matlab 7.1, but 'uchar' needed
    % for 7.2 and higher.
    case   2, [hdr.vol nitemsread] = fread(fp,inf,'uchar');
    case   4, [hdr.vol nitemsread] = fread(fp,inf,'short');
    case   8, [hdr.vol nitemsread] = fread(fp,inf,'int');
    case  16, [hdr.vol nitemsread] = fread(fp,inf,'float');
    case  64, [hdr.vol nitemsread] = fread(fp,inf,'double');
    case 512, [hdr.vol nitemsread] = fread(fp,inf,'ushort');
    case 768, [hdr.vol nitemsread] = fread(fp,inf,'uint');
    otherwise,
        fprintf('ERROR: data type %d not supported',hdr.datatype);
        hdr = [];
        return;
end

fclose(fp);


% Check that that many voxels were read in
if(nitemsread ~= nvoxels)
    fprintf('ERROR: %s, read in %d voxels, expected %d\n',...
        niftifile,nitemsread,nvoxels);
    hdr = [];
    return;
end

if(IsIco7)
    %fprintf('load_nifti: ico7 reshaping\n');
    hdr.dim(2) = 163842;
    hdr.dim(3) = 1;
    hdr.dim(4) = 1;
    dim = hdr.dim(2:end);
end

hdr.vol = reshape(hdr.vol, dim');
if(hdr.scl_slope ~= 0)
    % fprintf('Rescaling NIFTI: slope = %g, intercept = %g\n',...
    % hdr.scl_slope,hdr.scl_inter);
    %fprintf('    Good luck, this has never been tested ... \n');
    hdr.vol = hdr.vol * hdr.scl_slope  + hdr.scl_inter;
end

return;


function [vol, M, mr_parms, volsz] = load_mgh(fname,slices,frames,headeronly)
% [vol, M, mr_parms, volsz] = load_mgh(fname,<slices>,<frames>,<headeronly>)
%
% fname - path of the mgh file
%
% slices - list of one-based slice numbers to load. All
%   slices are loaded if slices is not specified, or
%   if slices is empty, or if slices(1) <= 0.
%
% frames - list of one-based frame numbers to load. All
%   frames are loaded if frames is not specified, or
%   if frames is empty, or if frames(1) <= 0.
%
% M is the 4x4 vox2ras transform such that
% y(i1,i2,i3), xyz1 = M*[i1 i2 i3 1] where the
% indices are 0-based. If the input has multiple frames,
% only the first frame is read.
%
% mr_parms = [tr flipangle te ti fov]
%
% volsz = size(vol). Helpful when using headeronly as vol is [].
%
% See also: save_mgh, vox2ras_0to1
%


%
% load_mgh.m
%
% Original Author: Bruce Fischl
% CVS Revision Info:
%    $Author: nicks $
%    $Date: 2011/03/02 00:04:12 $
%    $Revision: 1.20 $
%
% Copyright © 2011 The General Hospital Corporation (Boston, MA) "MGH"
%
% Terms and conditions for use, reproduction, distribution and contribution
% are found in the 'FreeSurfer Software License Agreement' contained
% in the file 'LICENSE' found in the FreeSurfer distribution, and here:
%
% https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
%
% Reporting: freesurfer@nmr.mgh.harvard.edu
%

vol = [];
M = [];
mr_parms = [];
volsz = [];

if(nargin < 1 | nargin > 4)
    msg = 'USAGE: [vol M] = load_mgh(fname,<slices>,<frames>,<headeronly>)';
    fprintf('%s',msg);
    return;
end

if(exist('slices')~=1) slices = []; end
if(isempty(slices)) slices = 0; end
if(slices(1) <= 0) slices = 0; end

if(exist('frames')~=1) frames = []; end
if(isempty(frames)) frames = 0; end
if(frames(1) <= 0) frames = 0; end

if(exist('headeronly')~=1) headeronly = 0; end

fid    = fopen(fname, 'rb', 'b') ;
if(fid == -1)
    fprintf('ERROR: could not open %s for reading\n',fname);
    return;
end
v       = fread(fid, 1, 'int') ;
if(isempty(v))
    fprintf('ERROR: problem reading fname\n');
end
ndim1   = fread(fid, 1, 'int') ;
ndim2   = fread(fid, 1, 'int') ;
ndim3   = fread(fid, 1, 'int') ;
nframes = fread(fid, 1, 'int') ;
type    = fread(fid, 1, 'int') ;
dof     = fread(fid, 1, 'int') ;

if(slices(1) > 0)
    ind = find(slices > ndim3);
    if(~isempty(ind))
        fprintf('ERROR: load_mgh: some slices exceed nslices\n');
        return;
    end
end

if(frames(1) > 0)
    ind = find(frames > nframes);
    if(~isempty(ind))
        fprintf('ERROR: load_mgh: some frames exceed nframes\n');
        return;
    end
end

UNUSED_SPACE_SIZE= 256;
USED_SPACE_SIZE = (3*4+4*3*4);  % space for ras transform

unused_space_size = UNUSED_SPACE_SIZE-2 ;
ras_good_flag = fread(fid, 1, 'short') ;
if (ras_good_flag)
    delta  = fread(fid, 3, 'float32') ;
    Mdc    = fread(fid, 9, 'float32') ;
    Mdc    = reshape(Mdc,[3 3]);
    Pxyz_c = fread(fid, 3, 'float32') ;
    
    D = diag(delta);
    
    Pcrs_c = [ndim1/2 ndim2/2 ndim3/2]'; % Should this be kept?
    
    Pxyz_0 = Pxyz_c - Mdc*D*Pcrs_c;
    
    M = [Mdc*D Pxyz_0;  ...
        0 0 0 1];
    ras_xform = [Mdc Pxyz_c; ...
        0 0 0 1];
    unused_space_size = unused_space_size - USED_SPACE_SIZE ;
end

fseek(fid, unused_space_size, 'cof') ;
nv = ndim1 * ndim2 * ndim3 * nframes;
volsz = [ndim1 ndim2 ndim3 nframes];

MRI_UCHAR =  0 ;
MRI_INT =    1 ;
MRI_LONG =   2 ;
MRI_FLOAT =  3 ;
MRI_SHORT =  4 ;
MRI_BITMAP = 5 ;

% Determine number of bytes per voxel
switch type
    case MRI_FLOAT,
        nbytespervox = 4;
    case MRI_UCHAR,
        nbytespervox = 1;
    case MRI_SHORT,
        nbytespervox = 2;
    case MRI_INT,
        nbytespervox = 4;
end

if(headeronly)
    fseek(fid,nv*nbytespervox,'cof');
    if(~feof(fid))
        [mr_parms count] = fread(fid,4,'float32');
        if(count ~= 4)
            fprintf('This file does not contain MRI parameters\n');
        end
    end
    fclose(fid);
    return;
end


%------------------ Read in the entire volume ----------------%
if(slices(1) <= 0 & frames(1) <= 0)
    switch type
        case MRI_FLOAT,
            vol = fread(fid, nv, 'float32') ;
        case MRI_UCHAR,
            vol = fread(fid, nv, 'uchar') ;
        case MRI_SHORT,
            vol = fread(fid, nv, 'short') ;
        case MRI_INT,
            vol = fread(fid, nv, 'int') ;
    end
    
    if(~feof(fid))
        [mr_parms count] = fread(fid,4,'float32');
        if(count ~= 4)
            fprintf('This file does not contain MRI parameters\n');
        end
    end
    fclose(fid) ;
    
    nread = prod(size(vol));
    if(nread ~= nv)
        fprintf('ERROR: tried to read %d, actually read %d\n',nv,nread);
        vol = [];
        return;
    end
    vol = reshape(vol,[ndim1 ndim2 ndim3 nframes]);
    
    return;
end

%----- only gets here if a subest of slices/frames are to be loaded ---------%


if(frames(1) <= 0) frames = [1:nframes]; end
if(slices(1) <= 0) slices = [1:ndim3]; end

nvslice = ndim1 * ndim2;
nvvol   = ndim1 * ndim2 * ndim3;
filepos0 = ftell(fid);
vol = zeros(ndim1,ndim2,length(slices),length(frames));
nthframe = 1;
for frame = frames
    
    nthslice = 1;
    for slice = slices
        filepos = ((frame-1)*nvvol + (slice-1)*nvslice)*nbytespervox + filepos0;
        fseek(fid,filepos,'bof');
        
        switch type
            case MRI_FLOAT,
                [tmpslice nread]  = fread(fid, nvslice, 'float32') ;
            case MRI_UCHAR,
                [tmpslice nread]  = fread(fid, nvslice, 'uchar') ;
            case MRI_SHORT,
                [tmpslice nread]  = fread(fid, nvslice, 'short') ;
            case MRI_INT,
                [tmpslice nread]  = fread(fid, nvslice, 'int') ;
        end
        
        if(nread ~= nvslice)
            fprintf('ERROR: load_mgh: reading slice %d, frame %d\n',slice,frame);
            fprintf('  tried to read %d, actually read %d\n',nvslice,nread);
            fclose(fid);
            return;
        end
        
        vol(:,:,nthslice,nthframe) = reshape(tmpslice,[ndim1 ndim2]);
        nthslice = nthslice + 1;
    end
    
    nthframe = nthframe + 1;
end

% seek to just beyond the last slice/frame %
filepos = (nframes*nvvol)*nbytespervox + filepos0;
fseek(fid,filepos,'bof');

if(~feof(fid))
    [mr_parms count] = fread(fid,5,'float32');
    if(count < 4)
        fprintf('This file does not contain MRI parameters\n');
    end
end

fclose(fid) ;

return;



function [fspec, fstem, fmt] = MRIfspec(fstring,checkdisk)
% [fspec fstem fmt] = MRIfspec(fstring,<checkdisk>)
%
% Determine the file specification (fspec), stem (fstem), and format
% (fmt) from the given file string.
%
% A specification is the name of a file as it could exist on disk. The
% specification has the form fstem.fmt, where fmt can be mgh, mgz,
% nii, nii.gz, img, bhdr. fstring can be either an fspec or fstem. 
%
% If fstring is an fspec, then the format is determined from the
% extension.
%
% If fstring is an fstem and checkdisk=1 or NOT present, then the
% format is determined by finding a file on disk named fstem.fmt. The
% formats are searched in the following order: mgh, mgz, bhdr, img,
% nii, nii.gz. The search order is only important when there are
% multiple files that would have met the criteria, then only the first
% one is chosen. If no such file is found, then empty strings are
% returned.
%


%
% MRIfspec.m
%
% Original Author: Doug Greve
% CVS Revision Info:
%    $Author: nicks $
%    $Date: 2011/03/02 00:04:12 $
%    $Revision: 1.8 $
%
% Copyright © 2011 The General Hospital Corporation (Boston, MA) "MGH"
%
% Terms and conditions for use, reproduction, distribution and contribution
% are found in the 'FreeSurfer Software License Agreement' contained
% in the file 'LICENSE' found in the FreeSurfer distribution, and here:
%
% https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
%
% Reporting: freesurfer@nmr.mgh.harvard.edu
%


fspec = [];
fstem = [];
fmt   = [];

if(nargin < 1 | nargin > 2)
  fprintf('[fspec fstem fmt] = MRIfspec(fstring,<checkdisk>)\n');
  return;
end

% If checkdisk is not passed, then do a check disk
if(~exist('checkdisk','var')) checkdisk = 1; end

inddot = max(findstr(fstring,'.'));
if(isempty(inddot))
  ext = ' ';
else
  ext = fstring(inddot+1:end);
end
    
switch(ext)
 case 'mgh',
  fspec = fstring;
  fmt = 'mgh';
  fstem = fstring(1:end-4);
  return;
 case 'mgz',
  fspec = fstring;
  fmt = 'mgz';
  fstem = fstring(1:end-4);
  return;
 case 'nii',
  fspec = fstring;
  fmt = 'nii';
  fstem = fstring(1:end-4);
  return;
 case 'gz',
  ind = findstr(fstring,'nii.gz');
  if(~isempty(ind))
    fspec = fstring;
    fmt = 'nii.gz';
    fstem = fstring(1:ind-2);
    return;
  end
 case 'img',
  fspec = fstring;
  fmt = 'img';
  fstem = fstring(1:end-4);
  return;
 case 'hdr',
  fspec = fstring;
  fmt = 'img';
  fstem = fstring(1:end-4);
  return;
 case 'bhdr',
  fspec = fstring;
  fmt = 'bhdr';
  fstem = fstring(1:end-5);
  return;
end

% If it gets here, then it cannot determine the format from an
% extension, so fstring could be a stem, so see if a file with
% stem.fmt exists. Order is imporant here.

% Do this only if checkdisk is on
if(~checkdisk) return; end

fstem = fstring;

fmt = 'mgh';
fspec = sprintf('%s.%s',fstring,fmt);
if(fast_fileexists(fspec)) return; end

fmt = 'mgz';
fspec = sprintf('%s.%s',fstring,fmt);
if(fast_fileexists(fspec)) return; end

fmt = 'bhdr';
fspec = sprintf('%s.%s',fstring,fmt);
if(fast_fileexists(fspec)) return; end

fmt = 'img';
fspec = sprintf('%s.%s',fstring,fmt);
if(fast_fileexists(fspec)) return; end

fmt = 'nii';
fspec = sprintf('%s.%s',fstring,fmt);
if(fast_fileexists(fspec)) return; end

fmt = 'nii.gz';
fspec = sprintf('%s.%s',fstring,fmt);
if(fast_fileexists(fspec)) return; end

% If it gets here, then could not determine format
fstem = [];
fmt = [];
fspec = [];

return;


function hdr = load_nifti_hdr(niftifile)
% hdr = load_nifti_hdr(niftifile)
%
% Changes units to mm and msec.
% Creates hdr.sform and hdr.qform with the matrices in them.
% Creates hdr.vox2ras based on sform if valid, then qform.
% Does not and will not handle compressed. Compression is handled
% in load_nifti.m, which calls load_nifti_hdr.m after any
% decompression. 
%
% Endianness is returned as hdr.endian, which is either 'l' or 'b'. 
% When opening again, use fp = fopen(niftifile,'r',hdr.endian);
%
% Handles data structures with more than 32k cols by
% reading hdr.glmin = ncols when hdr.dim(2) < 0. This
% is FreeSurfer specific, for handling surfaces.
%
% $Id: load_nifti_hdr.m,v 1.10.4.1 2016/08/02 21:03:47 greve Exp $


%
% load_nifti_hdr.m
%
% Original Author: Doug Greve
% CVS Revision Info:
%    $Author: greve $
%    $Date: 2016/08/02 21:03:47 $
%    $Revision: 1.10.4.1 $
%
% Copyright © 2011 The General Hospital Corporation (Boston, MA) "MGH"
%
% Terms and conditions for use, reproduction, distribution and contribution
% are found in the 'FreeSurfer Software License Agreement' contained
% in the file 'LICENSE' found in the FreeSurfer distribution, and here:
%
% https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
%
% Reporting: freesurfer@nmr.mgh.harvard.edu
%


hdr = [];

if(nargin ~= 1)
  fprintf('hdr = load_nifti_hdr(niftifile)\n');
  return;
end

% Try opening as big endian first
fp = fopen(niftifile,'r','b');
if(fp == -1) 
  fprintf('ERROR: could not read %s\n',niftifile);
  return;
end

hdr.sizeof_hdr  = fread(fp,1,'int');
if(hdr.sizeof_hdr ~= 348)
  fclose(fp);
  % Now try opening as little endian
  fp = fopen(niftifile,'r','l');
  hdr.sizeof_hdr  = fread(fp,1,'int');
  if(hdr.sizeof_hdr ~= 348)
    fclose(fp);
    fprintf('ERROR: %s: hdr size = %d, should be 348\n',...
	    niftifile,hdr.sizeof_hdr);
    hdr = [];
    return;
  end
  hdr.endian = 'l';
else
  hdr.endian = 'b';
end

hdr.data_type       = fscanf(fp,'%c',10);
hdr.db_name         = fscanf(fp,'%c',18);
hdr.extents         = fread(fp, 1,'int');
hdr.session_error   = fread(fp, 1,'short');
hdr.regular         = fread(fp, 1,'char');
hdr.dim_info        = fread(fp, 1,'char');
hdr.dim             = fread(fp, 8,'short');
hdr.intent_p1       = fread(fp, 1,'float');
hdr.intent_p2       = fread(fp, 1,'float');
hdr.intent_p3       = fread(fp, 1,'float');
hdr.intent_code     = fread(fp, 1,'short');
hdr.datatype        = fread(fp, 1,'short');
hdr.bitpix          = fread(fp, 1,'short');
hdr.slice_start     = fread(fp, 1,'short');
hdr.pixdim          = fread(fp, 8,'float'); % physical units
hdr.vox_offset      = fread(fp, 1,'float');
hdr.scl_slope       = fread(fp, 1,'float');
hdr.scl_inter       = fread(fp, 1,'float');
hdr.slice_end       = fread(fp, 1,'short');
hdr.slice_code      = fread(fp, 1,'char');
hdr.xyzt_units      = fread(fp, 1,'char');
hdr.cal_max         = fread(fp, 1,'float');
hdr.cal_min         = fread(fp, 1,'float');
hdr.slice_duration  = fread(fp, 1,'float');
hdr.toffset         = fread(fp, 1,'float');
hdr.glmax           = fread(fp, 1,'int');
hdr.glmin           = fread(fp, 1,'int');
hdr.descrip         = fscanf(fp,'%c',80);
hdr.aux_file        = fscanf(fp,'%c',24);
hdr.qform_code      = fread(fp, 1,'short');
hdr.sform_code      = fread(fp, 1,'short');
hdr.quatern_b       = fread(fp, 1,'float');
hdr.quatern_c       = fread(fp, 1,'float');
hdr.quatern_d       = fread(fp, 1,'float');
hdr.quatern_x       = fread(fp, 1,'float');
hdr.quatern_y       = fread(fp, 1,'float');
hdr.quatern_z       = fread(fp, 1,'float');
hdr.srow_x          = fread(fp, 4,'float');
hdr.srow_y          = fread(fp, 4,'float');
hdr.srow_z          = fread(fp, 4,'float');
hdr.intent_name     = fscanf(fp,'%c',16);
hdr.magic           = fscanf(fp,'%c',4);

fclose(fp);

% This is to accomodate structures with more than 32k cols
% FreeSurfer specific. See also mriio.c.
if(hdr.dim(2) < 0) 
  hdr.dim(2) = hdr.glmin; 
  hdr.glmin = 0; 
end

% look at xyz units and convert to mm if needed
xyzunits = bitand(hdr.xyzt_units,7); % 0x7
switch(xyzunits)
 case 1, xyzscale = 1000.000; % meters
 case 2, xyzscale =    1.000; % mm
 case 3, xyzscale =     .001; % microns
 otherwise, 
  fprintf('WARNING: xyz units code %d is unrecognized, assuming mm\n',xyzunits);
  xyzscale = 2; % just assume mm
end
hdr.pixdim(2:4) = hdr.pixdim(2:4) * xyzscale;
hdr.srow_x = hdr.srow_x * xyzscale;
hdr.srow_y = hdr.srow_y * xyzscale;
hdr.srow_z = hdr.srow_z * xyzscale;

% look at time units and convert to msec if needed
tunits = bitand(hdr.xyzt_units,3*16+8); % 0x38 
switch(tunits)
 case  8, tscale = 1000.000; % seconds
 case 16, tscale =    1.000; % msec
 case 32, tscale =     .001; % microsec
 otherwise,  tscale = 0; 
end
hdr.pixdim(5) = hdr.pixdim(5) * tscale;

% Change value in xyzt_units to reflect scale change
hdr.xyzt_units = bitor(2,16); % 2=mm, 16=msec

% Sform matrix
hdr.sform =  [hdr.srow_x'; 
	      hdr.srow_y'; 
	      hdr.srow_z';
	      0 0 0 1];

% Qform matrix - not quite sure how all this works,
% mainly just copied CH's code from mriio.c
b = hdr.quatern_b;
c = hdr.quatern_c;
d = hdr.quatern_d;
x = hdr.quatern_x;
y = hdr.quatern_y;
z = hdr.quatern_z;
a = 1.0 - (b*b + c*c + d*d);
if(abs(a) < 1.0e-7)
  a = 1.0 / sqrt(b*b + c*c + d*d);
  b = b*a;
  c = c*a;
  d = d*a;
  a = 0.0;
else
  a = sqrt(a);
end
r11 = a*a + b*b - c*c - d*d;
r12 = 2.0*b*c - 2.0*a*d;
r13 = 2.0*b*d + 2.0*a*c;
r21 = 2.0*b*c + 2.0*a*d;
r22 = a*a + c*c - b*b - d*d;
r23 = 2.0*c*d - 2.0*a*b;
r31 = 2.0*b*d - 2*a*c;
r32 = 2.0*c*d + 2*a*b;
r33 = a*a + d*d - c*c - b*b;
if(hdr.pixdim(1) < 0.0)
  r13 = -r13;
  r23 = -r23;
  r33 = -r33;
end
qMdc = [r11 r12 r13; r21 r22 r23; r31 r32 r33];
D = diag(hdr.pixdim(2:4));
P0 = [x y z]';
hdr.qform = [qMdc*D P0; 0 0 0 1];

if(hdr.sform_code ~= 0)
  % Use sform first
  hdr.vox2ras = hdr.sform;
elseif(hdr.qform_code ~= 0)
  % Then use qform first
  hdr.vox2ras = hdr.qform;
else
  fprintf('WARNING: neither sform or qform are valid in %s\n', ...
	  niftifile);
  D = diag(hdr.pixdim(2:4));
  P0 = [0 0 0]';
  hdr.vox2ras = [eye(3)*D P0; 0 0 0 1];
end

return;



function M1 = vox2ras_0to1(M0)
% M1 = vox2ras_0to1(M0)
%
% Converts a 0-based vox2ras matrix to 1-based, ie,
% Pxyz = M0*[c r s 1]' = M1*[(c+1) (r+1) (s+1) 1]'
%


%
% vox2ras_0to1.m
%
% Original Author: Doug Greve
% CVS Revision Info:
%    $Author: nicks $
%    $Date: 2011/03/02 00:04:13 $
%    $Revision: 1.3 $
%
% Copyright © 2011 The General Hospital Corporation (Boston, MA) "MGH"
%
% Terms and conditions for use, reproduction, distribution and contribution
% are found in the 'FreeSurfer Software License Agreement' contained
% in the file 'LICENSE' found in the FreeSurfer distribution, and here:
%
% https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
%
% Reporting: freesurfer@nmr.mgh.harvard.edu
%


M1 = [];

if(nargin ~= 1)
  fprintf('M1 = vox2ras_0to1(M0)\n');
  return;
end

Q = zeros(4);
Q(1:3,4) = ones(3,1);

M1 = inv(inv(M0)+Q);

return;




function T = vox2ras_tkreg(voldim, voxres)
% T = vox2ras_tkreg(voldim, voxres)
%   voldim = [ncols  nrows  nslices ...]
%   volres = [colres rowres sliceres ...]
%
% Computes the 0-based vox2ras transform of a volume
% compatible with the registration matrix produced
% by tkregister. May work with MINC xfm.
%


%
% vox2ras_tkreg.m
%
% Original Author: Doug Greve
% CVS Revision Info:
%    $Author: nicks $
%    $Date: 2011/03/02 00:04:13 $
%    $Revision: 1.4 $
%
% Copyright © 2011 The General Hospital Corporation (Boston, MA) "MGH"
%
% Terms and conditions for use, reproduction, distribution and contribution
% are found in the 'FreeSurfer Software License Agreement' contained
% in the file 'LICENSE' found in the FreeSurfer distribution, and here:
%
% https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
%
% Reporting: freesurfer@nmr.mgh.harvard.edu
%

T = [];

if(nargin ~= 2)
  fprintf('T = vox2ras_tkreg(voldim, voxres)\n');
  return;
end

T = zeros(4,4);
T(4,4) = 1;

T(1,1) = -voxres(1);
T(1,4) = voxres(1)*voldim(1)/2;

T(2,3) = voxres(3);
T(2,4) = -voxres(3)*voldim(3)/2;


T(3,2) = -voxres(2);
T(3,4) = voxres(2)*voldim(2)/2;


return;












