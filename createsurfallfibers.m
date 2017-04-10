%surface creation for all fibers of one subject

dirName = '/Users/lindseykitchell/Box Sync/fiberVolumes/110411/';
filelist = dir('*boolVol*.nii.gz');
namelist = {filelist.name};
subjname = '110411';

for i=1:numel(namelist)
    fiber_vol = niftiRead(strcat(dirName,namelist{i}));
    
    filename_split = strsplit(namelist{i},{'_','.nii.gz'});
    fibname = filename_split(1,7:(length(filename_split)-1));
    tracking = filename_split(1,3);
    surfname = strjoin(fibname,'_');
    surfname = strcat(subjname,'_', tracking,'_',surfname,'.off');
    
    [nodes, elems]=generateFibShapeSurface(fiber_vol, surfname);
end





