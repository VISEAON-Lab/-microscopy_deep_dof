clear all;
close all;
clc;

%% Folder Names Specified By The User
BaseFolderName                 = 'path/to/base/folder';
% 'name_of_specific_folder_name'
SpecificKernelFolderName       = 'ZEMAX_V2_2048';

%% Folder Names According To Pre Defined Structure
DataSetFolderName       = 'DataSet';
BaseKernelsFolderName   = 'Kernels';
BlurredImgFolderName    = 'BlurredImages';
SharpImgFolderName      = 'SharpImages';

BaseFolderNameForDocker = '/opt/project';

KernelFolderFullPath = fullfile(BaseFolderNameForDocker, DataSetFolderName, BaseKernelsFolderName, SpecificKernelFolderName, 'Stacks', 'imgNorm');

BlurredImgDir        = fullfile(DataSetFolderName, BlurredImgFolderName, SpecificKernelFolderName);
BlurDirFiles         = dir(fullfile(BaseFolderName, BlurredImgDir));

SharpImgFolder       = fullfile(DataSetFolderName, SharpImgFolderName);
SharpImgFiles        = dir(fullfile(BaseFolderName, SharpImgFolder, '*.png'));

%% Algorithm
FilePathSharp            = fullfile(BaseFolderNameForDocker, SharpImgFolder,{SharpImgFiles.name})';
TripletListForTrain      = cell(length(FilePathSharp), 3);
TripletListForTrain(:,1) = FilePathSharp;

for f = 3: (length(BlurDirFiles))
    currBlurDir              = BlurDirFiles(f).name;
    currBlurDirFiles         = dir(fullfile(BaseFolderName, BlurredImgDir, currBlurDir, '*.png')); 
    filePathBlur             = fullfile(BaseFolderNameForDocker, BlurredImgDir, currBlurDir, {currBlurDirFiles.name})'; % folder_name.name
    TripletListForTrain(:,2) = filePathBlur;
    
    currKernelWd             = regexp(currBlurDir,'\d+\.?\d*','match');
    kernelPath               = fullfile(KernelFolderFullPath, ['kernel_WD_', currKernelWd{:}, '.png']);
    TripletListForTrain(:,3) = {kernelPath};
    
    writecell(TripletListForTrain, fullfile(BaseFolderName, DataSetFolderName, [SpecificKernelFolderName, '.txt']), 'Delimiter','tab', 'WriteMode', 'append');% 'Delimiter','space'

end