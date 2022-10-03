close all;
clear all;
clc;

%% Folder Names Specified By The User
BaseFolderName               = 'path/to/base/folder';
% 'name_of_specific_folder_name'
SpecificZmaxKernelFolderName = 'ZEMAX_V2_2048';


%% Folder Names According To Pre Defined Structure
DataSetFolderName       = 'DataSet';
BaseKernelsFolderName   = 'Kernels';

KernelFullPathFolderName   = fullfile(BaseFolderName, DataSetFolderName, BaseKernelsFolderName, SpecificZmaxKernelFolderName);
KernelFullResFolder        = fullfile(KernelFullPathFolderName, 'FullRes');
StackSavePath              = fullfile(KernelFullPathFolderName, 'Stacks');

mkdir(fullfile(StackSavePath, 'matNorm'));
mkdir(fullfile(StackSavePath, 'imgNorm'));
mkdir(fullfile(StackSavePath, 'imgViz'));
mkdir(fullfile(StackSavePath, 'plotCross'));

KernelFullResFiles = natsortfiles(dir(KernelFullResFolder));

%%
load(fullfile(KernelFullPathFolderName, 'args.mat'));

fullResRows = 2047; % The size of the symmetric full resoultion kernel
numericalFactor = 6; % to avoid numerical factors

%% Algorithm
for wdIdx = 3 : length(KernelFullResFiles) -1
    currWdFolder = KernelFullResFiles(wdIdx).name;
    disp(currWdFolder);
    filePattern = fullfile(KernelFullResFolder, currWdFolder, '*.mat');
    focusSet = natsortfiles(dir(filePattern));
    kernel3D = zeros(fullResRows, fullResRows, length(focusSet));
    cntForCheckKerNum = 0;
    for focusIdx = 1 : length(focusSet)
        %checks that the kernel is not out of bound
        currentName = focusSet(focusIdx).name;
        wd = str2double(currentName(3:strfind(currentName, '_')-1));
        beginningOfDefocusIndex = strfind(currentName, 'defocus') + 7;
        endOfDefocusIndex = strfind(currentName, '.mat') - 1;
        defocus = str2double(currentName(beginningOfDefocusIndex:endOfDefocusIndex));
        if (wd + defocus < 77) || (wd + defocus > 92.510)
            continue
        end
        cntForCheckKerNum = cntForCheckKerNum +1;
        
        focusPath = fullfile(KernelFullResFolder, currWdFolder, focusSet(focusIdx).name);
        [tmpKernel, add, resizedRows] = perFocusPlaneKernel(focusPath, numericalFactor, fullResRows);
        kernel3D(ceil(1+add):floor(resizedRows+add), ceil(1+add):floor(resizedRows+add), focusIdx) = tmpKernel;    
    end
    fprintf('The number of kernels for wd %.3f is %d \n', wd, cntForCheckKerNum);
   
    kernel = mean(kernel3D, 3); %averaging all the kernels after resize to the same size

    kernel = imresize(kernel, [341 341]);
    kernel = kernel / sum(kernel(:));
    figure; plot(squeeze(kernel(floor((length(kernel)+1)/2), :)));
    
    save(fullfile(StackSavePath, 'matNorm', ['kernel_', currWdFolder, '.mat']), 'kernel');
    imwrite(im2uint8(kernel), fullfile(StackSavePath, 'imgNorm', ['kernel_', currWdFolder, '.png']));
    
    imwrite(kernel / max(kernel(:)), fullfile(StackSavePath, 'imgViz', ['kernel_', currWdFolder, '.png']));
    
    figure('visible', 'off');
    plot(kernel(floor(length(kernel) / 2 + 1), :));
    ylim([0 1.4e-2]);
    saveas(gcf, fullfile(StackSavePath, 'plotCross', ['crossSection_', currWdFolder, '.png']));
end
