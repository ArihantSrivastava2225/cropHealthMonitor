% MATLAB Script for Texture Feature Enhancement
%
% This script runs AFTER the Python preprocessing pipeline. It reads the
% 6-channel .mat files, calculates four GLCM texture features from the
% NIR band, and saves a new, 10-channel .mat file.

clear; clc; close all;
addpath(pwd); % Ensure functions in the same folder are accessible

disp('======================================================');
disp('Starting MATLAB Feature Enhancement Process...');
disp('======================================================');

% --- Configuration ---
% Define input and output directories relative to the 'src' folder
baseDir = fileparts(fileparts(mfilename('fullpath')));
processedDir = fullfile(baseDir, 'data', 'processed');
finalDir = fullfile(baseDir, 'data', 'final_matlab_enhanced');

if ~isfolder(finalDir), mkdir(finalDir); end

% Get the list of event folders from the processed directory
eventFolders = dir(processedDir);
eventFolders = eventFolders([eventFolders.isdir]); % Keep only directories
eventFolders = eventFolders(~ismember({eventFolders.name}, {'.', '..'}));

% --- Main Processing Loop ---
for i = 1:length(eventFolders)
    eventName = eventFolders(i).name;
    eventInputDir = fullfile(processedDir, eventName);
    eventOutputDir = fullfile(finalDir, eventName);
    
    if ~isfolder(eventOutputDir), mkdir(eventOutputDir); end
    
    fprintf('\n--- Processing Event: %s ---\n', eventName);
    
    matFiles = dir(fullfile(eventInputDir, '*.mat'));
    
    for j = 1:length(matFiles)
        matFileName = matFiles(j).name;
        fprintf('  - Processing file: %s\n', matFileName);
        
        % Load the 6-channel patch data from the Python script
        load(fullfile(eventInputDir, matFileName), 'patches'); % Loads the 'patches' variable
        
        % Get dimensions
        [numPatches, patchSize, ~, numChannels] = size(patches);
        
        % Pre-allocate a new array for the 10-channel data (6 original + 4 texture)
        enhancedPatches = zeros(numPatches, patchSize, patchSize, 10, 'single');
        
        % Copy the original 6 channels
        enhancedPatches(:,:,:,1:6) = patches;
        
        fprintf('    -> Calculating texture features for %d patches...\n', numPatches);
        
        % Loop through each patch to calculate texture features
        for p = 1:numPatches
            % Extract the NIR band (Channel 4)
            % Data shape is (num_patches, H, W, C)
            nir_patch = squeeze(patches(p, :, :, 4));
            
            % Rescale NIR band to 8-bit integer range [0, 255] for GLCM
            if max(nir_patch(:)) > min(nir_patch(:))
                nir_uint8 = uint8(255 * mat2gray(nir_patch));
            else
                nir_uint8 = zeros(size(nir_patch), 'uint8');
            end
            
            % Calculate Gray-Level Co-occurrence Matrix
            % We compute it for 4 directions (0, 45, 90, 135 degrees)
            % and then average the stats for rotational invariance.
            glcms = graycomatrix(nir_uint8, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
            
            % Calculate texture statistics from the GLCM
            stats = graycoprops(glcms, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
            
            % Average the stats across all directions
            avg_stats = mean(stats, 1);
            
            % Create a 2D "image" for each texture feature
            enhancedPatches(p, :, :, 7) = avg_stats.Contrast;
            enhancedPatches(p, :, :, 8) = avg_stats.Correlation;
            enhancedPatches(p, :, :, 9) = avg_stats.Energy;
            enhancedPatches(p, :, :, 10) = avg_stats.Homogeneity;
        end
        
        % Save the new 10-channel data to a new .mat file
        % The variable inside will still be named 'patches' for consistency
        patches = enhancedPatches;
        outputMatPath = fullfile(eventOutputDir, matFileName);
        save(outputMatPath, 'patches', '-v7.3');
        fprintf('    -> Saved 10-channel enhanced data to: %s\n', matFileName);
        
    end
end

disp('======================================================');
disp('MATLAB Feature Enhancement Complete!');
fprintf('Final 10-channel data is ready in: %s\n', finalDir);
disp('You can now upload the ''final_matlab_enhanced'' folder to Kaggle.');
