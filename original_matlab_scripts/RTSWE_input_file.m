        %% Input file for RTSWE estimates

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

% Location of txt file with generated data

