function [matlabdate_SN_vec, station_SWE_mat, SIERRA_WIDE_SWE_average, SIERRA_WIDE_vec] = RTSWE_prepare_data(file_SWE_SN, dir_station_SWE, save_path_CDEC)

disp('Preparing SN data...');
tic

 load(file_SWE_SN)
 
 year_data = 1984:2018;
 for y = 1:length(year_data)
     for d = 274:365
         [yy, mm, dd] = jd2cal(doy2jd(year_data(y), d));
         matlabdate_SN(y, d-273) = datenum(yy, mm, dd);
     end
     
     for d = 1:273
         [yy, mm, dd] = jd2cal(doy2jd(year_data(y)+1, d));
         matlabdate_SN(y, d+91) = datenum(yy, mm, dd);
     end
     
     matlabdate_SN(y, 365) = datenum(yy, mm, dd+1);
 end;
 
 matlabdate_SN = matlabdate_SN';
 matlabdate_SN_vec = matlabdate_SN(:);
 SIERRA_WIDE_vec = SIERRA_WIDE_SWE_vs_time(:);
 
 SIERRA_WIDE_SWE_average = nanmean(SIERRA_WIDE_SWE_vs_time, 2);

 toc
 
% Load sensor data 
 
 disp('Preparing sensor data...');
 tic
 date_today = num2str(round(now, 0));
 
 filename_station_mat = [dir_station_SWE '_' date_today]
 
 if exist([filename_station_mat '.mat'], 'file')
     load(filename_station_mat);
 else
     % call function to standardize sensor data
     
     dir_sens = dir([save_path_CDEC '*.csv']);
     station_SWE_mat = nan(length(matlabdate_SN_vec), length(dir_sens));
     
     for s = 1:length(dir_sens)
         
         if mod(s, 10) == 0
             disp(['Reading sensor ' num2str(s) ' out of ' num2str(length(dir_sens))]);
         end
         
         path_data = [save_path_CDEC dir_sens(s).name];
         [matdatenum, ~, ~, ~, values] = RTSWE_sensor_reader(path_data);
         [~, B, C] = intersect(matlabdate_SN(:), matdatenum);
         values(values<0) = NaN;
         %station_SWE(s).data = nan(size(matlabdate_SN_vec));
         %station_SWE(s).data(B) = values(C);
         station_SWE_mat(B, s) = values(C);
     end;
     
     save(filename_station_mat, 'station_SWE_mat');
 end;
 
station_SWE_mat(isnan(station_SWE_mat)) = 0;
toc