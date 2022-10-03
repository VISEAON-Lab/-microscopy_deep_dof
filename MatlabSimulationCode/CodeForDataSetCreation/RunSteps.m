close all;
clear all;
clc;
%%
% Author: Eden Sassoon
% Data of creation: 20.09.2020

%% User Definition
ZEMAX_PATH      = 'C:\Users\Administrator\Documents\Zemax';
% This is the .net serial application running commands (should be in the ZOAPI/ folder)
netHelperFile   = fullfile(ZEMAX_PATH, 'ZOS-API\User Analysis\ZOSAPI_NetHelper.dll'); 

systemFilesPath = 'D:\Naama\MicroscopeSimulation\ZEMAX_systemFiles';
% systemFileName  = 'Mutitoyo and Optotune in water - v3';
systemFileName  = 'Mutitoyo_Optotune_v2';
% expname         = 'V3_under_water_2048_full';
expname         = 'V2_2048_full';
n_underwatr = 4 /3;
n_vaccum = 1;
n = n_vaccum;

%% build args
args.systemFileName   = systemFileName;
args.n                = n;
args.NetHelper        = netHelperFile;
args.zmxfilepath      = fullfile(systemFilesPath, [systemFileName '.ZMX']);
results_path          = ['ZEMAX_resultsFullRange_' expname];
args.DOF              = 0.282 * n;
args.show             = false;
args.numerical_factor = 6; % to avoid numerical factors

mkdir(results_path);
save(fullfile(results_path,'args.mat') ,'args');

%%
% wd - working distance
initialWd = 92.792 * n;
wdNum = 57;
wdList = initialWd -args.DOF * (0:wdNum-1);

for i=1:length(wdList)
    wd = wdList(i);
    args.wd = wd;
    args.curr_save_path = fullfile(results_path, 'FullRes', ['WD_', num2str(wd, '%.3f')]);
    args.resized_save_path = fullfile(results_path, 'Resized', ['WD_', num2str(wd, '%.3f'), '_resized']);
    mkdir(args.curr_save_path);
    mkdir(args.resized_save_path);
    zmx_seq_app(args);
    disp(['finished ' num2str(100 * i / length(wdList)) '%']);
end


