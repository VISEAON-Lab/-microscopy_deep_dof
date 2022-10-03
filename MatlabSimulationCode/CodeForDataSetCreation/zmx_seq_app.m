function [ r ] = zmx_seq_app(args)

if ~exist('args', 'var')
    args = [];
end

% Initialize the OpticStudio connection
import ZOSAPI.*;
TheApplication = InitConnection(args);
if isempty(TheApplication)
    % failed to initialize a connection
    r = [];
else
    try
        r = BeginApplication(TheApplication, args);
        CleanupConnection(TheApplication);
    catch err
        CleanupConnection(TheApplication);
        rethrow(err);
    end
end
end

function [r] = BeginApplication(TheApplication, args)

import ZOSAPI.*;


    % Load a sample file
    TheSystem = TheApplication.PrimarySystem;
    
    %! [e15s01_m]
%     args.workpath

% 	bstat = TheSystem.LoadFile(System.String.Concat(TheApplication.SamplesDir,args.zmxfilepath), false);
    bstat = TheSystem.LoadFile(System.String.Concat(args.zmxfilepath), false);
%     if bstat
%         fprintf(['loading model from ',args.zmxfilepath,':\n']);
%     end
    if bstat
        fprintf('loading model ');
    end
        %! [e15s01_m]
    
    %! [e15s07_m]
    % remove all variables and add a F# solve on last surface radius
    TheLDE = TheSystem.LDE;
    Surf_0 = TheLDE.GetSurfaceAt(0);
    Surf_0.Thickness = args.wd;
    Surf_1 = TheLDE.GetSurfaceAt(1);
    Surf_1.Thickness = 0;
%     Surf_17 = TheLDE.GetSurfaceAt(17);
%     Surf_18 = TheLDE.GetSurfaceAt(18);
%     Surf_7 = TheLDE.GetSurfaceAt(7);
    
    
%     Surf_17.ThicknessCell.MakeSolveVariable();
%     Surf_18.ThicknessCell.MakeSolveVariable();
    
%     SampleFile = System.String.Concat( 'D:\Judith\MicroscopeDetails\Eden\Save_System\OptimizedFile', num2str(args.defocus),'.ZMX');
%     TheSystem.SaveAs(SampleFile);    
    
    %%Optimization%%%
    
    LocalOpt = TheSystem.Tools.OpenLocalOptimization();
    LocalOpt.Algorithm = ZOSAPI.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares;
    LocalOpt.Cycles = ZOSAPI.Tools.Optimization.OptimizationCycles.Automatic;
    LocalOpt.NumberOfCores = 12;
    LocalOpt.RunAndWaitForCompletion();
    LocalOpt.Close();
       
     
%     SampleFile = System.String.Concat('D:\Judith\MicroscopeDetails\Eden\Save_System\OptimizedFile_wd', [num2str(args.wd), '_defocus', num2str(args.defocus),'.ZMX']);
%     TheSystem.SaveAs(SampleFile);    
    
    for defocus_num = 0 : 9
        tic;
        %edit args
        dof = args.DOF*defocus_num;
        
        defocus = -dof;
%         defocus = dof;

        % Change defocus
        Surf_1 = TheLDE.GetSurfaceAt(1);
        Surf_1.Thickness = defocus;

        % Create PSF Cross Section analysis
        newWin = TheSystem.Analyses.New_FftPsfCrossSection();
        newWin_Settings = newWin.GetSettings();
    %     newWin_Settings.MaximumFrequency = 50;
        newWin_Settings.SampleSize = ZOSAPI.Analysis.SampleSizes.S_512x512;
%         newWin_Settings.RowCol = 256;
        newWin_Settings.PlotScale = 50;

       % Run Analysis & gets results
        newWin.ApplyAndWaitForCompletion();
        newWin_Results = newWin.GetResults();
        
        if args.show
            figure('position', [50, 150, 900, 600])
        else
            figure('position', [50, 150, 900, 600], 'visible','off')
        end
        hold on;
        grid on;

        dataSeries = newWin_Results.DataSeries;
        cc=lines(double(newWin_Results.NumberOfDataSeries));
        for gridN=1:newWin_Results.NumberOfDataSeries
            data = dataSeries(gridN);
            y = data.YData.Data.double;
            x = data.XData.Data.double;
            ylim([0,1]);
            plot(x,y(:,1),'-','color',cc(gridN,:));
    %         plot(x,y(:,2),':','color',cc(gridN,:));
        end
        %! [e04s05_m]

        title('FFT PSF Cross Section');
        xlabel('X-Position in (\mum)');
        ylabel('Relativ Irradiance');

        saveas(gcf, fullfile(args.curr_save_path, ['CrossSection_wd', num2str(args.wd), '_defocus', num2str(defocus), '.png']));
        %! [e15s07_m]

        % Create FFT PSF Analysis
        PSFWin = TheSystem.Analyses.New_FftPsf();
    %     PSFWin = ZOSAPI.Analysis.Psf.AS_FftPsf;
        PSFWin_Settings = PSFWin.GetSettings();
        PSFWin_Settings.SampleSize = ZOSAPI.Analysis.Settings.Psf.PsfSampling.PsfS_2048x2048;
        PSFWin_Settings.OutputSize = ZOSAPI.Analysis.Settings.Psf.PsfSampling.PsfS_2048x2048;
%         PSFWin_Settings.SampleSize = ZOSAPI.Analysis.Settings.Psf.PsfSampling;

       % Run Analysis & gets results
        PSFWin.ApplyAndWaitForCompletion();
        PSFWin_Results = PSFWin.GetResults();

    %     figure('position', [50, 150, 900, 600])
    %     hold on;
    %     grid on;

        PSFdataGrids = PSFWin_Results.DataGrids;
        cc3D=lines(double(PSFWin_Results.NumberOfDataGrids));
        for grid3D=1:PSFWin_Results.NumberOfDataGrids
            PSFdata = PSFdataGrids(gridN); 
            y2 = PSFdata.ValueData.Data.double;
            Dx = PSFdata.Dx; %data_spacing
    %         x2 = PSFdata.XData.Data.double;
    %         z2 = PSFdata.ZData.Data.double;
    %         plot(x2,y2,z2(:,1),'-','color',cc3D(grid3D,:, :));
    %         plot(x,y(:,2),':','color',cc(gridN,:));
        end
%         res_f = (Dx/5.5) * args.numerical_factor;
%         k_res = imresize(y2, res_f);
        
        if args.show
            figure();
        else
            figure('visible','off');
        end
        imagesc(y2);
        name = ['wd', num2str(args.wd, '%.3f'), '_defocus', num2str(defocus, '%.3f')];
        saveas(gcf, fullfile(args.curr_save_path, [name, '.png']));
        save(fullfile(args.curr_save_path, [name, '.mat']), 'y2', 'Dx');
%         imwrite(k_res, fullfile(args.resized_save_path, [name, '_Dx', num2str(Dx, '%.3f'), '.png']));
%         save(fullfile(args.resized_save_path, [name, '_Dx', num2str(Dx, '%.3f'), '.mat']), 'k_res')
        
    %! [e04s05_m]
        
    close all;
    end

    r = [];
end
% 

function app = InitConnection(args)

import System.Reflection.*;

% Find the installed version of OpticStudio.

% This method assumes the helper dll is in the .m file directory.
% p = mfilename('fullpath');
% [path] = fileparts(p);
% p = strcat(path, '\', 'ZOSAPI_NetHelper.dll' );
% NET.addAssembly(p);

% This uses a hard-coded path to OpticStudio
NET.addAssembly(args.NetHelper);

success = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize();
% Note -- uncomment the following line to use a custom initialization path
% success = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize('C:\Program Files\OpticStudio\');
if success == 1
    LogMessage(strcat('Found OpticStudio at: ', char(ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory())));
else
    app = [];
    return;
end
% 
% Now load the ZOS-API assemblies
NET.addAssembly(AssemblyName('ZOSAPI_Interfaces'));
NET.addAssembly(AssemblyName('ZOSAPI'));

% Create the initial connection class
TheConnection = ZOSAPI.ZOSAPI_Connection();

% Attempt to create a Standalone connection

% NOTE - if this fails with a message like 'Unable to load one or more of
% the requested types', it is usually caused by try to connect to a 32-bit
% version of OpticStudio from a 64-bit version of MATLAB (or vice-versa).
% This is an issue with how MATLAB interfaces with .NET, and the only
% current workaround is to use 32- or 64-bit versions of both applications.
app = TheConnection.CreateNewApplication();
if isempty(app)
   HandleError('An unknown connection error occurred!');
end
if ~app.IsValidLicenseForAPI
    HandleError('License check failed!');
    app = [];
end

end

function LogMessage(msg)
disp(msg);
end

function HandleError(error)
ME = MXException(error);
throw(ME);
end

function  CleanupConnection(TheApplication)
% Note - this will close down the connection.

% If you want to keep the application open, you should skip this step
% and store the instance somewhere instead.
TheApplication.CloseApplication();
end


