%% Test wrapper for basin-wide SWE real-time model

% Stage to run
stages = [1 2 3];

% Path to code/data
path_name = '/Users/gcortes/Dropbox/projects/ongoing/project_real_time_SWE/';

% Path where to save the downlaoded sensor data
save_path_CDEC = '/Users/gcortes/Dropbox/projects/ongoing/project_real_time_SWE/data/CDEC_sensor_data/SWE/';

% Source of SWE data, 1 for courses, 2 for sensors
source = 2;

% Number of sensor, 82 for SWE
number = 82;

% Location of SIERRA wide data
file_SWE_SN = '/Users/gcortes/Dropbox/projects/ongoing/project_real_time_SWE/data/SN_SWE_data/SIERRA_WIDE_SWE_vs_time.mat';

% Location of stored sensor data
dir_station_SWE = ['/Users/gcortes/Dropbox/projects/ongoing/project_real_time_SWE/data/station_SWE'];
addpath(genpath(path_name));
addpath(genpath('/Users/gcortes/Documents/MATLAB/'))

years = 1984:2015;

basins = {'American'; 'Carson'; 'Cosumnes'; 'Feather'; 'Kaweah'; 'Kern'; 'Kings'; 'Merced'; 'Mokelumne'; 'Mono'; ...
    'Owens'; 'San_Joaquin'; 'Stanislaus'; 'Tahoe' ; 'Truckee'; 'Tule'; 'Tuolumne' ;'upper_sacramento'; 'walker'; 'yuba'};

% 2 - Prepare sensor data with nearby stations

if ~isempty(find(stages == 2, 1))
    disp(['Preparing the data...']);
    [matlabdate_SN_vec, station_SWE_mat, ~, ~] = RTSWE_prepare_data(file_SWE_SN, dir_station_SWE, save_path_CDEC);
end

station_SWE_mat(station_SWE_mat>1000) = 0;
%%
for b = 1:length(basins)
    
    basin_name = basins{b}
    
    % Organize data
    
    load(['/Volumes/elqui_hd2/PROJECTS/SWE_REANALYSIS/SNEVADA/PEAK_SWE_DATA/Post_peak_SWE_' basin_name '_90m_outputs_merged.mat'])
    basin_WIDE_vec_y = reshape(BASIN_AVE_SWE, 365, 32);
    basin_WIDE_SWE_average = nanmean(basin_WIDE_vec_y, 2);
    area_basin = length(APRIL01_SWE)*90*90/1000000000;    % Run the model
    
    if ~isempty(find(stages == 3, 1))
        [date_data, model_data, model_SWE] = ...
            RTSWE_predict(matlabdate_SN_vec, station_SWE_mat, basin_WIDE_SWE_average, ...
            BASIN_AVE_SWE(:), 0);
    end
    
    % Generate plot
    
    lag = 3;
    today_date = date_data.date_pred(date_data.date_pred == round(now,0));
    [yy, mm, dd] = datevec(today_date - lag);
    doy_today = jd2doy(cal2jd(yy, mm, dd));
    if doy_today<274
        doy_today = doy_today + 91;
    else
        doy_today = 365 - doy_today;
    end;
    
    status_date = [num2str(yy) '-' num2str(mm) '-' num2str(dd)];
    
    % Generate plot data
    subplot(4,5,b);
    area(basin_WIDE_SWE_average*area_basin, 'FaceColor', [.8 .8 .8], 'LineStyle', 'none')
    ind_today = find(date_data.date_pred == round(now,0));
    hold on;
    plot(area_basin*model_data.data_pred((ind_today - doy_today - lag):( ind_today - lag)), 'LineWidth', 2)
    xlim([50 365]);
    %ylim([0 .5]);
    grid on;
    set(gca, 'FontSize', 14);
    
     status_volume = model_data.data_pred(date_data.date_pred == round(now,0) - lag);
    status_perc = status_volume./basin_WIDE_SWE_average(doy_today)*100;
    
    title([basin_name ' ' num2str(round(status_perc)) '%']);
    box on;
    if b == 19
        
         xlabel(['Basin storage ' status_date]);
    end
    if b == 20
        ylabel(['Total storage [km^3]']);
        xlabel('Day of the Water Year');
    end;
    %legend '1984-2016 average' '2017-18 Water Year' %'1990-91 Water Year'
    %print(['/Users/gcortes/Dropbox/projects/ongoing/project_real_time_SWE/figures/' basin_name '_' status_date],'-dpng');
    
    % Tweet data
    
   
end;
