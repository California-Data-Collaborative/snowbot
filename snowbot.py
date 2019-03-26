#wrapper functions for the snowbot tasks

def fetchData():

	import pandas as pd
	import urllib
	from datetime import datetime, date, timedelta
	import boto3
	import os
	import io
	import re

	def getStationDetails():
	    df = pd.read_csv('data/stationNames.csv', thousands=',')
	    return(df[['ID', 'Longitude', 'Latitude', 'ElevationFeet']])

	bucket = 'cadc-snowbot'
	s3_client = boto3.client('s3')
	s3_resource = boto3.resource('s3')

	stationIDs = getStationDetails().ID
	first_date = '1980-07-02'
	today = str(date.today())
	yesterday = str(date.today()-timedelta(1))

	def listAll():
	    tmp = s3_client.list_objects(Bucket='cadc-snowbot')
	    if 'Contents' in tmp:
	        return [i['Key'] for i in tmp['Contents']]
	    else:
	        return None

	def removeFile(filename=None):
	    if filename != None:
	        return False
	    else:
	        s3_client.delete_object(Bucket='cadc-snowbot', Key=filename)
	        return True

	def updateHistorical(sens=3, start_date=first_date, end_date=yesterday):
	    st = '3LK'
	    url = 'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={}&SensorNums={}&dur_code=D&Start={}&End={}'.format(st,sens,start_date,end_date)

	    df_raw = pd.read_csv(url)
	    df_raw['station_id'] = st
	    df = pd.DataFrame()

	    for st in stationIDs[1:]:
	        url = 'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={}&SensorNums={}&dur_code=D&Start={}&End={}'.format(st,sens,start_date,end_date)
	        tmp = pd.read_csv(url)

	        df_raw = df_raw.append(tmp)

	        tmp['date'] = [datetime.strptime(i[:8], '%Y%m%d').date() \
	                       for i in tmp['DATE TIME']]
	        tmp = tmp[['date', 'VALUE']]
	        tmp.columns = ['date', st]
	        tmp.set_index('date')

	        df = tmp if df.shape[0]==0 else df.merge(tmp, how='outer')

	    if((start_date == first_date) & (end_date == yesterday)):
	        backupData(df, df_raw, sens)

	    #df.to_csv("station_data.csv")
	    #df_raw.to_csv("station_data_raw.csv")

	    return df, df_raw

	def backupData(df, df_raw, sens):
	    fname = "{}_0{}".format(today.replace('-',''),sens)
	    fnameBackup = "backup_{}.csv".format(fname)
	    fnameRaw = "raw_{}.csv".format(fname)

	    df.to_csv("data/{}".format(fnameBackup))
	    df_raw.to_csv("data/{}".format(fnameRaw))

	    s3_resource.Object(bucket,fnameBackup).upload_file(Filename='data/{}'.format(fnameBackup))

	    s3_resource.Object(bucket,fnameRaw).upload_file(Filename='data/{}'.format(fnameRaw))

	    if os.path.exists("data/{}".format(fnameBackup)):
	        os.remove("data/{}".format(fnameBackup))
	    if os.path.exists("data/{}".format(fnameRaw)):
	        os.remove("data/{}".format(fnameRaw))

	def getLastFile(sens='03', prefix='backup', download=False):
	    r = re.compile("%s.*%s.csv"%(prefix, sens))
	    f = [i for i in listAll() if r.match(i)]
	    file = max(f)
	    if download:
	        s3_resource.Object(bucket, file).download_file(Filename='data/{}'.format(file))
	    else:
	        obj = s3_client.get_object(Bucket=bucket, Key=file)#.download_file(Filename=f'data/{file}')
	        return(pd.read_csv(io.BytesIO(obj['Body'].read())))

	def getUrl(st, sens, start, end):
	    return('http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={}&SensorNums={}&dur_code=D&Start={}&End={}'.format(st,sens,today,today))

	def getToday(start_date = yesterday):
	    df, df_raw = updateHistorical(3, start_date, yesterday)
	    return(df)

	# try:
	data1, data2 = updateHistorical(3)
	# except:
	# 	data1 = getLastFile('0{}'.format(sens), 'backup')
    #     data2 = getLastFile('0{}'.format(sens), 'raw')
        # filename=max(os.listdir('data/raw*'))
        # processRaw(filename)
		# return(data1, data2)
	# data1, data2 = dailyUpdate(sens=3)
	data1.to_csv("station_level_data.csv")
	data2.to_csv("station_level_data_raw.csv")

def krigingscript():

	from pykrige.ok3d import OrdinaryKriging3D
	from pykrige.uk3d import UniversalKriging3D

	from pykrige.rk import Krige
	from pykrige.compat import GridSearchCV
	from numpy import typecodes
	import numpy
	import pandas as pd

	def kriging_per_row(all_data_daily_slice):

	    param_dict3d = {"method":["ordinary3d", "universal3d"],"variogram_model": ["linear", "power", "gaussian", "spherical"]}

	    estimator = GridSearchCV(Krige(), param_dict3d, verbose=False)
	    interpolated_values = pd.DataFrame()

	    for index,row_under_observation in all_data_daily_slice.iterrows():
	        row_under_observation = pd.DataFrame(row_under_observation)
	        transposed_row = row_under_observation.T

	              #merge using station ids as indices
	        snow_amt_with_locn = all_data_daily_slice.merge(row_under_observation,left_index = True, right_index = True)
	        snow_amt_with_locn.rename(columns = {index : 'snow_adj_inches'} , inplace = True)
	        snow_amt_with_locn['snow_adj_mters'] = snow_amt_with_locn['snow_adj_inches'] * 0.0254

	  #containing non null values
	        snow_amt_with_locn_notnull = snow_amt_with_locn.dropna()
	    #print(snow_amt_with_locn_notnull.shape)

	  #containing null values
	        snow_amount_null = snow_amt_with_locn[snow_amt_with_locn['snow_adj_inches'].isnull() == True]
	        snow_amount_null.drop(['snow_adj_mters'],axis=1 , inplace = True)


	  # perform grid search to identify the good fitting variogram
	        if (snow_amt_with_locn_notnull.shape[0] != 0 and snow_amt_with_locn_notnull.shape[0] != 1):
	            lons=numpy.array(snow_amt_with_locn_notnull['Longitude_Metres'])
	            lons = lons[~numpy.isnan(lons)]

	            lats=numpy.array(snow_amt_with_locn_notnull['Latiitude_Metres'])
	            lats = lats[~numpy.isnan(lats)]
	            elev=numpy.array(snow_amt_with_locn_notnull['ElevationRelative'])
	            snow_amount =numpy.array(snow_amt_with_locn_notnull['snow_adj_mters'])
	      # count the number of zeros in snow_amount
	      #print(snow_amount)

	            zero_count = (snow_amount == 0.0).sum()
	            zero_count_fraction = (zero_count / snow_amount.shape[0])


	        if numpy.all(snow_amount == 0.0) or zero_count_fraction >= 0.9:
	            predicted_Values = numpy.zeros(snow_amount_null.shape[0])
	            predicted_snow_values = pd.DataFrame(predicted_Values,index =snow_amount_null.index.values.tolist() , columns = ['snow_adj_mters'])


	        else:
	            lons_null=numpy.array(snow_amount_null['Longitude_Metres'])
	            lats_null=numpy.array(snow_amount_null['Latiitude_Metres'])
	            elev_null=numpy.array(snow_amount_null['ElevationRelative'])
	            X = numpy.array(snow_amt_with_locn_notnull[['Longitude_Metres','Latiitude_Metres', 'ElevationRelative']])
	            y = numpy.array(snow_amt_with_locn_notnull['snow_adj_mters'])
	            estimator = GridSearchCV(Krige(), param_dict3d, verbose=False)


	        try:
	            estimator.fit(X=X, y=y, verbose=False)
	        # find the best kriging technique:
	            if hasattr(estimator, 'best_score_'):
	                print('best_score R²={}'.format(round(estimator.best_score_,2)))
	                print('best_params = ', estimator.best_params_)


	            if(estimator.best_params_['method'] == 'universal3d' ):
	                ok3d = UniversalKriging3D(lons, lats, elev, snow_amount, variogram_model=estimator.best_params_['variogram_model'])
	                predicted_Values, variance_locn = ok3d.execute('points',  lons_null,lats_null,elev_null)

	            else:
	                sim3d = OrdinaryKriging3D(lons, lats, elev, snow_amount, variogram_model=estimator.best_params_['variogram_model'])
	                predicted_Values, variance_locn = sim3d.execute('points',  lons_null,lats_null,elev_null)


	        except ValueError:
	            sim3d = OrdinaryKriging3D(lons, lats, elev, snow_amount, variogram_model='gaussian')
	            predicted_Values, variance_locn = sim3d.execute('points',  lons_null,lats_null,elev_null)


	            predicted_snow_values = pd.DataFrame(predicted_Values,index =snow_amount_null.index.values.tolist() , columns = ['snow_adj_mters'])

	            interplated_df = pd.merge(predicted_snow_values,snow_amount_null,left_index = True, right_index = True)

	            final_row = pd.concat([snow_amt_with_locn_notnull,interplated_df])

	            final_row_snow = final_row[['snow_adj_mters']]
	            final_row_snow_transpose = final_row_snow.T
	            final_row_snow_transpose = final_row_snow_transpose[stn_data.ID.values.tolist()]
	            interpolated_values = interpolated_values.append(final_row_snow_transpose)

	    else:
	        last_row = interpolated_values.tail(1)
	        interpolated_values = interpolated_values.append(last_row)



	    return interpolated_values
    
	today = str(date.today())
	sens = 3
	fname = "{}_0{}".format(today.replace('-',''),sens)
	all_data_daily_slice = 'data/backup_{}.csv".format(fname) 
	data = kriging_per_row(all_data_daily_slice)
	data.to_csv("interpolated_data.csv")


def production_script():
	import pandas as pd
	import numpy as np
	import zipfile
	import pickle
	from datetime import datetime, date, timedelta
	from sklearn.metrics import mean_squared_error
	import matplotlib.pyplot as plt
	import xgboost as xgb
	from scipy.stats import pearsonr

	def read_swe(url_path):
	    '''function to read in and format the raw swe data (y data)'''
	    swe_vol = pd.read_csv(url_path, header=None, names=['date', 'area', 'vol'])
	    swe_vol['date'] = pd.to_datetime(swe_vol['date'])
	    swe_vol.set_index('date', inplace=True)
	    swe_vol.drop(columns=['area'], axis=1, inplace=True)
	    swe_vol = pd.DataFrame(swe_vol)

	    return swe_vol


	def read_file(file_path):
	    '''Function to read in daily x data'''
	    station = pd.read_csv(file_path)
	    station.dropna(axis=1,how='all',inplace=True)
	    station.replace('---', '0', inplace=True)        
	    station['date'] = pd.to_datetime(station['date'])
	    station = station.sort_values(by='date')
	    station.set_index('date', inplace=True)  # put date in the index
	    station = station[station.index > '1984-09-29']
        # removes days where there is no y-data
	    cols = station.columns
	    station[cols] = station[cols].apply(pd.to_numeric, errors='coerce')
	    station = station.apply(lambda row: row.fillna(row.mean()), axis=1)
	    try:
	        station.drop(columns=['Unnamed: 0'], axis=1, inplace=True)  # drop non-station columns
	    except:
	        pass

	    return station

	def merge_data(df1, df2):
	    """merge station and swe estimate data"""
	    swe = pd.merge(left=df1, right=df2, left_index=True, right_index=True)
	    swe.dropna(axis=1, how='all', inplace=True)

	    x = swe.iloc[:, 1:]
	    y = swe.iloc[:, :1]

	    x_array = x.values

	    return swe, x, y, x_array

	def create_lags(x, n_in=1, n_out=1, dropnan=True):

	    n_vars = 1 if type(x) is list else x.shape[1]
	    df = pd.DataFrame(x)
	    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
	    for i in range(n_in, 0, -1):
	        cols.append(df.shift(i))
	        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
	    for i in range(0, n_out):
	        cols.append(df.shift(-i))
	        if i == 0:
	            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
	        else:
	            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
	    agg = pd.concat(cols, axis=1)
	    agg.columns = names
    # drop rows with NaN values
	    if dropnan:
	        agg.dropna(inplace=True)
 
	    return agg

	def train_test_split(x, y):
	    y.reset_index(inplace=True)
	    y_filter = y.iloc[2:]
	    xy_lag = pd.merge(left=y_filter, right=x, how='left', left_index=True, right_index=True)
	    xy_lag.set_index('date', inplace=True)
        
	    last_data = datetime.date(2018, 8, 29)
	    today = datetime.date.today()
	    predict_range = today - last_data
	    predict_range = round(age.total_seconds() / 86400)

	    # split into three data frames: train, test, and predict
	    train = xy_lag.iloc[:round(len(xy_lag) * 0.65)]
	    test = xy_lag.iloc[round(len(xy_lag) * 0.65):-(predict_range)]
	    predict = xy_lag.iloc[-(predict_range):]  # predict is last 125 rows where there is no Y-data

	    train.dropna(subset=['vol'], inplace=True)
	    test.dropna(subset=['vol'], inplace=True)

	    # split into X and Y for each of train, test, and predict
	    y_train = train['vol']
	    x_train = train.drop(['vol'],axis=1)

	    y_test = test['vol']
	    x_test = test.drop(['vol'],axis=1)

	    y_predict = predict['vol']
	    x_predict = predict.drop(['vol'],axis=1)

	    return y_train, x_train, y_test, x_test, y_predict, x_predict, train, test, predict


	def model(x_train, x_test, x_predict, y_test, y_train):
	    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree=0.3, 
                                  learning_rate=0.05,max_depth=6, n_estimators=150,seed=13)
	    xg_reg.set_params(min_child_weight=5)
	    xg_reg.set_params(gamma=5)
	    xg_reg.set_params(reg_alpha=0)
	    xg_reg.set_params(reg_lambda=1.5)

	    xg_reg.fit(x_train,y_train)
	    train_preds = xg_reg.predict(x_train)
	    test_preds = xg_reg.predict(x_test)
	    predict_preds = xg_reg.predict(x_predict)
	    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
	    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds
                                               ))
	    return train_preds, test_preds, predict_preds, test_rmse, train_rmse

	def vis1(today, train, train_preds, test, test_preds, predict, predict_preds, y_train, y_test, train_rmse, test_rmse):
	    # join back to dates for graphing

	    train_graph = pd.merge(left=pd.DataFrame(train.index), right=pd.DataFrame(train_preds), how='left',
	                           left_index=True, right_index=True)

	    test_graph = pd.merge(left=pd.DataFrame(test.index), right=pd.DataFrame(test_preds), how='left',
	                          left_index=True, right_index=True)

	    pred_graph = pd.merge(left=pd.DataFrame(predict.index), right=pd.DataFrame(predict_preds), how='left',
	                          left_index=True, right_index=True)

	    train_graph.set_index('date', inplace=True)
	    test_graph.set_index('date', inplace=True)
	    pred_graph.set_index('date', inplace=True)

	    # convert training data to data frames for graphing
	    y_train = pd.DataFrame(y_train)
	    y_test = pd.DataFrame(y_test)

	    # Plot long time series

	    fig = plt.figure(figsize=(20, 10))
	    ax = fig.add_subplot(111)

	    ax.plot(train_graph[0], color='g', linestyle='--', label='Train Pred')
	    ax.plot(y_train['vol'], color='b', alpha=0.5, label='Train Act.')

	    ax.plot(test_graph[0], color='c', linestyle='--', label='Test Pred')
	    ax.plot(y_test['vol'], color='b', alpha=0.5, label='Test Act.')

	    ax.plot(pred_graph[0], color='m', label='Forecast')

	    ax.set_xlabel("Date")
	    ax.set_ylabel("SWE Volume by Day")
	    ax.set_title("Daily SWE Estimates Across the Sierras, 1984-2018, Actual vs. Estimates and Projection")
	    plt.legend()

	    ax.text('1984-1-1', 47, "RMSE of the train set: {}".format(round(train_rmse, 4)))
	    ax.text('1984-1-1', 45, "RMSE of the test set: {}".format(round(test_rmse, 4)))
        
	    ax.set_ylim(0,50)

	    font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 14}

	    plt.rc('font', **font)

	    plt.savefig('SWE Time Series {}.png'.format(today))

	def vis2(today, train, test, predict, train_preds, test_preds, predict_preds, y_train, y_test):

	    # join back to dates for graphing

	    train_graph = pd.merge(left=pd.DataFrame(train.index), right=pd.DataFrame(train_preds), how='left',
	                           left_index=True, right_index=True)

	    test_graph = pd.merge(left=pd.DataFrame(test.index), right=pd.DataFrame(test_preds), how='left',
	                          left_index=True, right_index=True)

	    pred_graph = pd.merge(left=pd.DataFrame(predict.index), right=pd.DataFrame(predict_preds), how='left',
	                          left_index=True, right_index=True)

	    train_graph.set_index('date', inplace=True)
	    test_graph.set_index('date', inplace=True)
	    pred_graph.set_index('date', inplace=True)

	    # convert training data to data frames for graphing
	    y_train = pd.DataFrame(y_train)
	    y_test = pd.DataFrame(y_test)

	    # scatter plot to show comparison of predicted vs actual on the training and test data sets

	    fig = plt.figure(figsize=(15, 15))
	    ax = fig.add_subplot(122)

	    ax.scatter(y_test['vol'], test_graph[0], color='b')

	    ax1 = fig.add_subplot(121)

	    ax1.scatter(y_train['vol'], train_graph[0], color='g')

	    ax.set_xlabel("Actual")
	    ax.set_ylabel("Predicted")
	    ax.set_title("Daily SWE Across the Sierras, 1984-2017, Actual vs. Predicted (Test set)")

	    ax1.set_xlabel("Actual")
	    ax1.set_ylabel("Predicted")
	    ax1.set_title("Daily SWE Across the Sierras, 1984-2017, Actual vs. Predicted (Train set)")

	    ax.text(1,37,"Test correlation: {}".format(round(y_test['vol'].corr(test_graph[0]), 2)))
	    ax1.text(1,37,"Train correlation: {}".format(round(y_train['vol'].corr(train_graph[0]), 3)))

	    ax.set_ylim(0,40)
	    ax.set_xlim(0,40)
        
	    ax1.set_ylim(0,40)
	    ax1.set_xlim(0,40)
        
	    font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 10}

	    plt.rc('font', **font)

	    plt.savefig("SWE Correlation Plots {}.png".format(today))
        
	def vis3(today, train, test, predict, train_preds, test_preds, predict_preds, y_train, y_test):

	    train_graph = pd.merge(left=pd.DataFrame(train.index), right=pd.DataFrame(train_preds), how='left',
	                           left_index=True, right_index=True)

	    test_graph = pd.merge(left=pd.DataFrame(test.index), right=pd.DataFrame(test_preds), how='left',
	                          left_index=True, right_index=True)

	    pred_graph = pd.merge(left=pd.DataFrame(predict.index), right=pd.DataFrame(predict_preds), how='left',
	                          left_index=True, right_index=True)

	    train_graph.set_index('date', inplace=True)
	    test_graph.set_index('date', inplace=True)
	    pred_graph.set_index('date', inplace=True)

	    # convert training data to data frames for graphing
	    y_train = pd.DataFrame(y_train)
	    y_test = pd.DataFrame(y_test)

	    # prep historical data for graphing

	    total_swe = pd.concat([y_train, y_test])

	    total_swe.reset_index(inplace=True)

	    # extract year and day of year from full date
	    total_swe['year'] = total_swe['date'].dt.year
	    total_swe['doy'] = total_swe['date'].dt.dayofyear

	    # set up select series for graph

	    avg_swe_series = total_swe[total_swe['date'] < '2017-10-01'].groupby('doy')['vol'].mean()
	    select_year_series = total_swe[(total_swe['date'] < '1991-10-01') & (total_swe['date'] > '1990-09-30')]

	    # format data frames

	    avg_swe_series = pd.DataFrame(avg_swe_series)
	    select_year_series = select_year_series[['doy', 'vol']]

	    avg_swe_series.reset_index(inplace=True)

	    # calculate water day from calendar day

	    avg_swe_series['water_day'] = [x - 273 if x >= 273 else x + 92 for x in avg_swe_series['doy']]
	    select_year_series['water_day'] = [x - 273 if x >= 273 else x + 92 for x in select_year_series['doy']]

	    avg_swe_series = avg_swe_series.iloc[:, 1:]
	    select_year_series = select_year_series.iloc[:, 1:]

	    actuals_series = pd.merge(select_year_series, avg_swe_series, how='left', left_on='water_day', right_on='water_day')

	    actuals_series['std_upper'] = actuals_series['vol_y'] + actuals_series['vol_y'].std()
	    actuals_series['std_lower'] = actuals_series['vol_y'] - actuals_series['vol_y'].std()

	    # adjust the prediction data

	    pred_graph.reset_index(inplace=True)

	    pred_series = pred_graph.iloc[33:]  # just takes dates for the beginning of the water year

	    # extract year and day of year from full date
	    pred_series['year'] = pred_series['date'].dt.year
	    pred_series['doy'] = pred_series['date'].dt.dayofyear

	    pred_series['water_day'] = [x - 273 if x >= 273 else x + 92 for x in pred_series['doy']]

	    pred_series = pred_series[['water_day', 0]]

	    #get day level standard deviation
	    total_swe['water_day'] = [x-273 if x >= 273 else x+92 for x in total_swe['doy']]

	    swe_agg = total_swe.groupby('water_day')['vol'].agg(['mean','std'])

	    #get upper and lower standard deviation bounds
	    swe_agg['std_upper'] = swe_agg['mean'] + swe_agg['std']
	    swe_agg['std_lower'] =  swe_agg['mean'] - swe_agg['std']

	    #resent index so I can use water day in the charting
	    swe_agg.reset_index(inplace=True)

	    fig = plt.figure(figsize=(20,10))
	    ax = fig.add_subplot(111)

	    ax.plot(actuals_series['water_day'],actuals_series['vol_x'],color='c',label='1990-91 Water Year',linewidth=3)
	    ax.plot(actuals_series['water_day'],actuals_series['vol_y'],color='b',linestyle='--',alpha=0.5,label='1984-2016 Avg. Water Year',linewidth=3)
	    ax.plot(pred_series['water_day'],pred_series[0],color='g',label='2018-19 Water Year Forecast',linewidth=3)

	    ax.set_xlabel("Day of the Water Year")
	    ax.set_ylabel("Sierra Nevada Total Storage [km3]")
	    ax.set_title("SWE Daily Estimate Comparison")
	    ax.set_ylim(0.15,50)
	    ax.set_xlim(0,365)

	    plt.fill_between(swe_agg['water_day'],swe_agg['std_lower'], swe_agg['std_upper'], facecolor='lightgray',label='1984-2016 +/- 1 std.')


	    plt.legend()

	    ax.text(1,48,"Updated as of: {}".format(today))
	    ax.text(1,46,"Today's Forecast: {} [km3]".format(pred_series[0].iloc[-1:].values))

	    font = {'family':'normal','weight':'bold','size': 16}

	    plt.text(0.05,-5,"Comment: Today's forecast is {}% of the historical average".format((pred_series[0].iloc[-1:].values / actuals_series['vol_y'][len(pred_series) -1]) * 100))
	    plt.rc('font', **font)

	    plt.savefig("Daily Water Year Graph_{}.png".format(today))

        
	today = str(date.today())
	sens = 3
	fname = "{}_0{}".format(today.replace('-',''),sens)
	file = "data/backup_{}.csv".format(fname)     
	url_path = 'https://s3-us-west-2.amazonaws.com/cawater-public/swe/pred_SWE.txt'

	swe_vol = read_swe(url_path)
	station = read_file(file)
	swe, x, y, x_array = merge_data(swe_vol, station)
	x_lag = create_lags(x_array, n_in=2, n_out=1)
	y_train, x_train, y_test, x_test, y_predict, x_predict, train, test, predict = train_test_split(x_lag, y)
	train_preds, test_preds, predict_preds, test_rmse, train_rmse = model(x_train, x_test,
                                                                          x_predict, y_test, y_train)

	vis1(today, train, train_preds, test, test_preds, predict, predict_preds, y_train, y_test, train_rmse, test_rmse)
	vis2(today, train, test, predict, train_preds, test_preds, predict_preds, y_train, y_test)
	vis3(today, train, test, predict, train_preds, test_preds, predict_preds, y_train, y_test)


def twitter_post(cons_key,cons_secret,access_token,at_secret):
	auth = tweepy.OAuthHandler(cons_key,cons_secret)
	auth.set_access_token(access_token,at_secret)
	api = tweepy.API(auth)
	list_of_files = glob.glob('path/*.png')
	latest_file = max(list_of_files, key=os.path.getctime)
	print(latest_file)

	image = open(latest_files, 'rb')
	message = 'placeholder'
	response = twitter.upload_media(media=image)
	media_id = [response['media_id']]
	twitter.update_status(status=message, media_ids=media_id)