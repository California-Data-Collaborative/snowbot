function RTSWE_wrapper(input_file)

run(input_file);
%%
addpath(genpath(path_name));


%% Real time estimation wrapper

%% 1 - Download the required CDEC data and extract the data into the format

if ~isempty(find(stages == 1, 1))
    disp(['Running CDEC download']);
    if ~exist([save_path_CDEC 'data_downloaded_' num2str(round(now, 0))], 'file')
        RTSWE_CDEC_retriever(source, path_name, number, save_path_CDEC);
        fid = fopen([save_path_CDEC 'data_downloaded_' num2str(round(now, 0))], 'w');
        fclose(fid);
    else
        disp(['Data already downloaded today']);
    end
end

%% FUTURE - Cleaning of the data for the ML methods


%% 2 - Prepare the data
addpath(genpath('/Users/gcortes/Documents/MATLAB/'));
if ~isempty(find(stages == 2, 1))
    disp(['Preparing the data...']);
    [matlabdate_SN_vec, station_SWE_mat, SIERRA_WIDE_SWE_average, SIERRA_WIDE_vec] = RTSWE_prepare_data(file_SWE_SN, dir_station_SWE, save_path_CDEC);
end

station_SWE_mat(station_SWE_mat>1000) = 0;

SIERRA_WIDE_vec_y = reshape(SIERRA_WIDE_vec, 365, 32);

%% 3 - Run the model

if ~isempty(find(stages == 3, 1))
    [date_data, model_data, model_SWE] = RTSWE_predict(matlabdate_SN_vec, station_SWE_mat, SIERRA_WIDE_SWE_average, SIERRA_WIDE_vec, 0);
end

%% 4 - Generate output data in adequate format for CaDC

fid = fopen([path_name 'pred_SWE_' num2str(round(now, 0)) '.txt'], 'w');

for r = 1:length(date_data.date_obs)
    if mod(r, 1000) == 0
        disp(['Writing day ' num2str(r) ' out of ' num2str(length(date_data.date_obs))]);
    end
    [yy, mm, dd] = datevec(date_data.date_obs(r));
    
    date_val = [num2str(yy) '-' num2str(mm) '-' num2str(dd)];
    fprintf(fid, [date_val ',whole-sierra,' num2str(model_data.data_obs(r)) '\n']);
end

for r = 1:length(date_data.date_pred)
    if mod(r, 100) == 0
        disp(['Writing day ' num2str(r) ' out of ' num2str(length(date_data.date_pred))]);
    end
    [yy, mm, dd] = datevec(date_data.date_pred(r));
    
    date_val = [num2str(yy) '-' num2str(mm) '-' num2str(dd)];
    fprintf(fid, [date_val ', whole-sierra,' num2str(model_data.data_pred(r)) '\n']);
end
fclose(fid);

awsPutCommand = ['/Users/gcortes/Library/Python/2.7/bin/aws s3 cp ' path_name 'pred_SWE_' num2str(round(now, 0)) '.txt s3://cawater-public/swe/pred_SWE.txt'];
system(awsPutCommand)

%% Generate plot

lag = 1;
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
close all

area(SIERRA_WIDE_SWE_average, 'FaceColor', [.8 .8 .8], 'LineStyle', 'none')
ind_today = find(date_data.date_pred == round(now,0));
hold on;
%plot(SIERRA_WIDE_vec_y(:, 7), 'LineWidth',2, 'Color',[1 0.3 0.3])
plot(SIERRA_WIDE_vec_y(:, 31), 'LineWidth',2, 'Color',[0 0 0])
plot(model_data.data_pred((ind_today - doy_today - lag):( ind_today - lag)), 'LineWidth', 2, 'Color',[0.3 0.3 1])
xlim([50 365]);
grid on;
set(gca, 'FontSize', 14);
title(status_date);
box on;
ylabel('Sierra Nevada total storage [km^3]');
xlabel('Day of the Water Year');
legend '1984-2016 average'  '1990-91 Water Year' '2017-18 Water Year'
print(['/Users/gcortes/Dropbox/projects/ongoing/project_real_time_SWE/figures/' status_date],'-dpng');

% Tweet data

status_volume = model_data.data_pred(date_data.date_pred == round(now,0) - lag);
status_perc = status_volume./SIERRA_WIDE_SWE_average(doy_today)*100;

addpath(genpath('/Users/gcortes/Documents/MATLAB/'));

credentials.ConsumerKey = 'GEma54Uj8od5TfgAlvCYRocmh';
credentials.ConsumerSecret = 'byWoJX2sUbtOtYyZPBLNRGuOcRQvMGjYJpji6G2fQAqpIkPyXh';
credentials.AccessToken = '968656486204284928-AsEooRchyNIWmht4OQR2bXegRT4yz9Z';
credentials.AccessTokenSecret = '9OohC1VJEcg1XhbQrQ1msGgx1VF5YJiiBDaiT3fJDxBAV';

%tw = twitty(credentials);
status_update = ['Sierra Nevada SWE as of ' status_date ': ' num2str(round(status_volume,2)) ' km^3. This is ' num2str(round(status_perc, 1)) '% of 1984-2017 average for this date. Model by @gcortes. #Snow #CAWater #Drought']
%S = tw.updateStatus(status_update); 


