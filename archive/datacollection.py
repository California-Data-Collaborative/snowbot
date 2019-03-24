import datetime
import pandas as pd
import urllib
#collection of daily 82 sensor level dat



#pull in daily data for all the stations
def daily_collection(main_df):

	''' 

	pull in the data daily from the cdec website

	'''

	# to do task: getting data from the database
	station_name_dataframe = pd.read_csv('data/stationNames.csv')
	today = str(datetime.date.today())
    sensor_number='82'

	for station in station_name_dataframe['ID'].values.tolist():
        sensor_url = 'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=' + station + '&SensorNums=' + sensor_number +'&dur_code=D&Start='+ today +'&End=' + today
        
        # read the url contents and redirect it to station files
        #urllib.request.urlretrieve(url, station+'.csv')
        stn_df = pd.read_csv(sensor_url)
        stn_filename = station + '.csv'
        stn_df.to_csv(stn_filename)

#concat all the files into a singe dataframe ; read the station id, snow_adj_amount and _date
	all_stns_df = pd.concat([pd.read_csv(file + '.csv' , usecols = ['STATION_ID' , 'DATE TIME' , 'VALUE' ]) for file in station_name_dataframe['ID'].values.tolist() ],axis=0)
	all_stns_df.rename(columns = {'DATE TIME' : 'date'}, inplace = True)
	all_stns_df.date = pd.to_datetime(all_stns_df.date)
	all_stns_df.date = all_stns_df.date.dt.date
	all_stns_df_pivot = all_stns_df.pivot(index='date', columns='STATION_ID', values='VALUE')
	all_stns_df_pivot.columns.name = None
	all_stns_df_pivot.reset_index(inplace=True)

	#concat to the main dataframe
	updated_df = pd.concat([main_df,all_stns_df_pivot],axis = 0)

	#to do task 2 : pushing data to database
	updated_df.to_csv('alldata.csv')
