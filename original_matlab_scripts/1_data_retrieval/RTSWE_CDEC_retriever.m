function RTSWE_CDEC_retriever(source, path_name, number, save_path_CDEC)

% Script to retrieve the data from the CDEC sensors and store them in the
% dropbox folder

addpath(genpath(path_name))

load stations_new
load stations_mg

if source == 1
    for i = 1:size(courses)
        aux = char(courses(i));
        filename = strcat([save_path_CDEC 'courses/' aux '.csv']);
         urlwrite(strcat(['http://cdec.water.ca.gov/cgi-progs/snowQuery?course_num='...
             aux '&month=%28All%29&start_date=&end_date=&data_wish=Raw+Data']), filename);
    end
end

if number == 30 
    sensors = temp;
end;
if number == 26 
    sensors = solar_rad;
end;
if number == 29 
    sensors = net_rad;
end;
if number == 45 
    sensors = precip;
end;
if number == 12 
    sensors = (strtrim(rel_hum));
end;
        
if source == 2
    for i = 1:size(sensors) 
        
        if mod(i, 10) == 0
            disp(['Downloading sensor ' num2str(i) ' out of ' num2str(length(sensors))]);
        end
        
        aux = char(sensors(i,:));
        
        if number == 30 
             filename = strcat([save_path_CDEC 'sensors/temp_day/' aux '.csv']);
        end;
        
        if number == 4
            filename = strcat([save_path_CDEC 'sensors/temp_day_average/' aux '.csv']);
        end;
        
        if number == 45
            filename = strcat([save_path_CDEC 'sensors/precip/' aux '.csv']);
        end;
        if number == 18
            filename = strcat([save_path_CDEC 'sensors/snow_depth/' aux '.csv']);
        end;
        
        if number == 82
            filename = strcat([save_path_CDEC aux '.csv']);
        end;
        
        if number == 26
            filename=strcat([save_path_CDEC 'sensors/solar_rad/' aux '.csv']);
        end;
        
        if number == 29
            filename=strcat([save_path_CDEC 'sensors/net_rad/' aux '.csv']);
        end;
        
        if number == 12
            filename=strcat([save_path_CDEC 'sensors/rel_hum_day/' aux '.csv']);
        end;
        urlwrite(strcat(['http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=' aux '&SensorNums=' num2str(number) '&dur_code=D&Start=1980-01-01&End=Now']), filename);
%         urlwrite(strcat(['http://cdec.water.ca.gov/cgi-progs/queryCSV?station_id=' aux '&sensor_num=' num2str(number) '&dur_code=D&start_date=01%2F01%2F1980&end_date=Now&data_wish=View+CSV+Data']), filename);
%         disp(strcat(['http://cdec.water.ca.gov/cgi-progs/queryCSV?station_id=' aux '&sensor_num=' num2str(number) '&dur_code=D&start_date=01%2F01%2F1980&end_date=Now&data_wish=View+CSV+Data']))
    end
    
end




