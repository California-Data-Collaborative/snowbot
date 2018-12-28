
function [matdatenum, year, month, day, values] = RTSWE_sensor_reader(NameFile)

%reads data from sensors downloaded using CDEC_retriever.m. The main output
%is the matlab date and values.
% written by Gonzalo Cortes May 2012
% Manu just edit to use textscan ... seem faster and cleaner also hours are
% important to output.

% Example:
% NameFile ='/Volumes/hydro2_hd2/DATA/CDEC/sensors/solar_rad/BKK.csv'
% [year,month,day,hr,min,values]=CDEC_sensor_reader_MG(NameFile)

fid = fopen(NameFile);
data = textscan(fid, '%s %s %s %s %s %s %s %s %s', 'headerlines', 2, 'TreatAsEmpty', '---', 'delimiter', ',');
fclose(fid);
yrmmdd = char(data{5});
empty_entries = find(strcmp(data{7},'---'));

if ~isempty(yrmmdd)
    year = str2num(yrmmdd(:, 1:4));
else
    year = [];
end

if ~isempty(year)
    month = str2num(yrmmdd(:,6:7));
    day = str2num(yrmmdd(:,9:10)); clear yrmmdd
    % hrmin = char(data{2});
    %  hr = str2num(hrmin(:,1:2));
    %min = str2num(hrmin(:,3:4)); clear hrmin
    values = str2double(data{7});
    values(empty_entries) = NaN;
    matdatenum = datenum([year month day]);
    clear data
else
    matdatenum = []; year = []; month = []; day = []; hr = []; min = []; values = [];
end
return

% counter=0;
% while ~feof(fid)
%     tline=fgetl(fid);
%
%     if strcmp(tline(1),'<')==1;
%         disp('not a valid file, check CDEC website to see if sensor is available');
%         break;
%     end;
%
%     %the next set of ifs looks for the beginning of the actual data
%     if strcmp(tline(1),'"')==1;
%         tline=fgetl(fid);
%         if strcmp(tline(5),',')==1;
%             tline=fgetl(fid);
%         end;
%     end;
%     counter=counter+1;
%     year(counter)=str2double(tline(1:4));
%     month(counter)=str2double(tline(5:6));
%     day(counter)=str2double(tline(7:8));
%     matlabdate(counter)=datenum(year(counter),month(counter),day(counter));
%     values(counter)=str2double(tline(15:end));
% end;
%
%
%
%
%
% if counter==0;
%     year=0;
%     month=0;
%     day=0;
%     matlabdate=0;
%     values=0;
% end;



