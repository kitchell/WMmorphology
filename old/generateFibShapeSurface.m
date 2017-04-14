function[newestnodes, newelement]=generateFibShapeSurface(fiberBoolNifti, filename)
%takes the boolean nifti object from generateBinaryFibVolume as its input.
%Truthfully, though it only really needs the .data field.  We may need to
%transfer information over about any resizing that gets done in order to
% make sure that the output to the printer is actually "life size".  It may
% be as simple as making sure that every node coordinate is converted back
% in to mm space.  Also, I'm not including any other parameter inputs at
% the moment, but this can be fixed later

%extracts the data we are interested in
boolMatrixVersionSmooth=fiberBoolNifti.data;
filename = filename{1};

%sets value for later padding.  We ingnore the part of the NIFTI volume
%that doesn't include any information because the iso2mesh code can't
%handle large files.
padval=1;
%finds xyz coordinates of non-zero entries in the nifti.data object
[i,j,k]=ind2sub(size(boolMatrixVersionSmooth),find(boolMatrixVersionSmooth>0));
%finds coordinates of bounding box
xmin=min(i);
xmax=max(i);
ymin=min(j);
ymax=max(j);
zmin=min(k);
zmax=max(k);

%check for boundary less than 1
xmin=max((xmin-padval), 1);
xmax=max((xmax+padval), 1);
ymin=max((ymin-padval), 1);
ymax=max((ymax+padval), 1);
zmin=max((zmin-padval), 1);
zmax=max((zmax+padval), 1);

%cuts out the unused parts of the new volume and puts that information in a
%new object
BoundedVolume=boolMatrixVersionSmooth(xmin:xmax,ymin:ymax,zmin:zmax);

%setting parameters for the iso2mesh code
opt.maxnode=1000000;
opt.radbound=.5;
opt.distbound=.5;
%opt=1.5


%we changed the vol2surf code to default to repairing and to have a default
%max node of 100000
fprintf('creating surface for %s \n', filename)
[node,elem,regions,holes]=v2s(uint8(BoundedVolume),[1],opt,'cgalsurf');

%It does this by default?
%[newnode,newelement]=meshcheckrepair(node,elem);

%reorients the normals so the faces all face outward
disp('reorienting normals')
[newnode,newelement]=surfreorient(node,elem);

%was at .5, but did not work.
%[newnode1,newelement1]=remeshsurf(newnode,newelement,.8);

%determines which nodes are connected, needed for smoothing
[conn, connum,count]=meshconn(newelement(:,1:3),length(newnode));

%smoothes the surface using a lowpass method

fprintf('smoothing the surface %s \n', filename)
newestnodes=smoothsurf(newnode,[],conn,50,.2,'lowpass',.5);

%plotmesh(newestnodes,newelement)

fprintf('saving %s file \n',filename)
saveoff(newestnodes, newelement, filename);
end