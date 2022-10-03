close all;
clear all;
clc;

%% Set Data Set Type
TransferImgToLinear = true;
if TransferImgToLinear
    DeGamma = 2.2;
else 
    DeGamma = 1;
end

%% Folder Names Specified By The User
BaseFolderName           = 'path/to/base/folder';
% 'name_of_specific_folder_name'
SpecificKernelFolderName = 'ZEMAX_V2_2048';

%% Folder Names According To Pre Defined Structure
DataSetFolderName       = 'DataSet';
BaseKernelsFolderName   = 'Kernels';
BlurredImgFolderName    = 'BlurredImages';
SharpImgFolderName      = 'SharpImages';

KernelFolderFullPath    = fullfile(BaseFolderName, DataSetFolderName, BaseKernelsFolderName, SpecificKernelFolderName);
BlurredImgSaveDir       = fullfile(BaseFolderName, DataSetFolderName, BlurredImgFolderName ,SpecificKernelFolderName);

SharpImgFolder          = fullfile(BaseFolderName, DataSetFolderName, SharpImgFolderName);
SharpImgFiles           = dir(fullfile(SharpImgFolder, '*.png'));

KernelFilesDir          = fullfile(KernelFolderFullPath, 'Stacks', 'matNorm');
kernelsFiles            = dir(fullfile(KernelFilesDir, '*.mat'));

%% Algorithm
textprogressbar('Blurring images: ');

for img_idx = 1 : length(SharpImgFiles)
    img_name = SharpImgFiles(img_idx).name;
    curr_img = im2double(imread(fullfile(SharpImgFolder, img_name)));
    curr_img_g = gpuArray(curr_img);
    curr_img_g = curr_img_g .^ DeGamma; 
    
    for k = 1 : length(kernelsFiles)
        curr_k_name = kernelsFiles(k).name;
        [~, Stackname, ~] = fileparts(fullfile(KernelFilesDir, curr_k_name));
        folder_name = Stackname(strfind(Stackname, 'kernel_')+9:end);
        curr_norm_save_dir = fullfile(BlurredImgSaveDir, ['blur', folder_name, '']);
        if img_idx == 1
            mkdir(curr_norm_save_dir);
        end
        load(fullfile(KernelFilesDir, curr_k_name));
        kernel_g = gpuArray(kernel);           
        blur_img = imfilter(curr_img_g, kernel_g, 'conv');
        blur_img = gather(blur_img);
        imwrite(blur_img, fullfile(curr_norm_save_dir, img_name));
    end 
    textprogressbar(100 * img_idx / length(SharpImgFiles));
end    
textprogressbar('done');