
# SWE PREDICTION PRODUCTION MODEL

# Import Library:

import pandas as pd
import numpy as np
import os
import datetime
import schedule
import time
import zipfile
import pickle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Function Library;


def read_swe(url):
    '''function to read in and format the raw swe data (y data)'''

    swe_vol = pd.read_csv(url, header=None, names=['date', 'area', 'vol'])
    swe_vol['date'] = pd.to_datetime(swe_vol['date'])
    swe_vol.set_index('date', inplace=True)
    swe_vol.drop(columns=['area'], axis=1, inplace=True)

    return pd.DataFrame(swe_vol)


def read_file(file_path, direct):
    '''Function to read in daily x data'''
    if os.path.exists(os.getcwd() + '/' + file_path) == True:
        station = pd.read_csv(file_path)
    else:
        zip_file = zipfile.Zipfile(file_path, 'r')
        zip_file.extractall(direct)
        station = pd.read_csv(file_path)

    station['date'] = pd.to_datetime(station['date'])
    station = station.sort_values(by='date')
    station.set_index('date', inplace=True)  # put date in the index
    station = station[station.index > '1984-09-29']  # removes days where there is no y-data
    station.replace('---', '0', inplace=True)
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
    """Frame a time series as a supervised learning dataset.Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning."""

    n_vars = 1 if type(x) is list else x.shape[1]
    df = pd.DataFrame(x)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
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

    # split into three data frames: train, test, and predict
    train = xy_lag.iloc[:round(len(xy_lag) * 0.65)]
    test = xy_lag.iloc[round(len(xy_lag) * 0.65):-125]
    predict = xy_lag.iloc[-125:]  # predict is last 125 rows where there is no Y-data

    train.dropna(subset=['vol'], inplace=True)
    test.dropna(subset=['vol'], inplace=True)

    # split into X and Y for each of train, test, and predict
    y_train = train['vol']
    x_train = train.iloc[:, 1:]

    y_test = test['vol']
    x_test = test.iloc[:, 1:]

    y_predict = predict['vol']
    x_predict = predict.iloc[:, 1:]

    return y_train, x_train, y_test, x_test, y_predict, x_predict, train, test, predict


def model(xgb_model, x_train, x_test, x_predict, y_test, y_train):
    xgb = pickle.load(open(xgb_model, "rb"))

    train_preds = xgb.predict(x_train)
    test_preds = xgb.predict(x_test)
    predict_preds = xgb.predict(x_predict)

    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))

    return train_preds, test_preds, predict_preds, test_rmse, train_rmse


def vis1(train, train_preds, test, test_preds, predict, predict_preds, y_train, y_test, train_rmse, test_rmse):
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

    ax.text('1984-1-1', 37, "RMSE of the train set: {}".format(round(train_rmse, 4)))
    ax.text('1984-1-1', 35, "RMSE of the test set: {}".format(round(test_rmse, 4)))

    plt.savefig('SWE Time Series.png')

    return train_graph, test_graph, pred_graph


def vis2(train, test, predict, train_preds, test_preds, predict_preds, y_train, y_test):

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

    fig = plt.figure(figsize=(15, 10))
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

    ax.annotate("Test correlation: {}".format(round(y_test['vol'].corr(test_graph[0]), 2),
                                              xy=(1, 28), xytext=(0, 30)))
    ax1.annotate("Train correlation: {}".format(round(y_train['vol'].corr(train_graph[0]), 3),
                                                xy=(1, 28), xytext=(0, 37.5)))

    plt.savefig("SWE Correlation Plots.png")


def vis3(train, test, predict, train_preds, test_preds, predict_preds, y_train, y_test):

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

    avg_swe_series['water_day'] = [x - 273 if x > 273 else x + 92 for x in avg_swe_series['doy']]
    select_year_series['water_day'] = [x - 273 if x > 273 else x + 92 for x in select_year_series['doy']]

    avg_swe_series = avg_swe_series.iloc[:, 1:]
    select_year_series = select_year_series.iloc[:, 1:]

    actuals_series = pd.merge(select_year_series, avg_swe_series, how='left', left_on='water_day', right_on='water_day')

    actuals_series['std_upper'] = actuals_series['vol_y'] + actuals_series['vol_y'].std()
    actuals_series['std_lower'] = actuals_series['vol_y'] - actuals_series['vol_y'].std()

    # adjust the prediction data

    pred_graph.reset_index(inplace=True)

    pred_series = pred_graph.iloc[-91:]  # just takes dates for the beginning of the water year

    # extract year and day of year from full date
    pred_series['year'] = pred_series['date'].dt.year
    pred_series['doy'] = pred_series['date'].dt.dayofyear

    pred_series['water_day'] = [x - 273 if x > 273 else x + 92 for x in pred_series['doy']]

    pred_series = pred_series[['water_day', 0]]

    fig = plt.figure(figsize=(25, 8))
    ax = fig.add_subplot(111)

    ax.plot(actuals_series['water_day'], actuals_series['vol_x'], color='c', label='1990-91 Water Year', linewidth=3)
    ax.plot(actuals_series['water_day'], actuals_series['vol_y'], color='b', linestyle='--', alpha=0.5,
            label='1984-2016 Avg. Water Year', linewidth=3)
    ax.plot(pred_series['water_day'], pred_series[0], color='g', label='2018-19 Water Year Forecast', linewidth=3)

    ax.set_xlabel("Day of the Water Year")
    ax.set_ylabel("Sierra Nevada Total Storage [km3]")
    ax.set_title("SWE Daily Estimate Comparison")
    ax.set_ylim(-1, 30)
    ax.set_xlim(0, 365)

    plt.fill_between(actuals_series['water_day'], actuals_series['std_lower'], actuals_series['std_upper'],
                     facecolor='lightgray', label='1984-2016 +/- 1 std.')

    plt.legend()

    ax.annotate("Updated as of: {}".format(datetime.datetime.now().strftime("%Y-%m-%d")), xy=(1, 28), xytext=(1.5, 28))
    ax.annotate("Today's Forecast: {} [km3]".format(pred_series[0].iloc[-1:].values), xy=(1, 28), xytext=(1.5, 26))

    plt.savefig("Daily Water Year Graph.png")


url_path = 'https://s3-us-west-2.amazonaws.com/cawater-public/swe/pred_SWE.txt'

file = 'fulldataset.csv'

xgb_pickle = "swe_xgb.pickle.dat"


def production_job():

    swe_vol = read_swe(url_path)
    station = read_file(file)
    swe, x, y, x_array = merge_data(swe_vol, station)
    x_lag = create_lags(x_array, n_in=2, n_out=1)
    train, test, predict, y_train, x_train, y_test, x_test, y_predict, x_predict = train_test_split(x_lag, y)
    train_preds, test_preds, predict_preds, test_rmse, train_rmse = model(xgb_pickle, x_train, x_test,
                                                                          x_predict, y_test, y_train)

    vis1(train, train_preds, test, test_preds, predict, predict_preds, y_train, y_test, train_rmse, test_rmse)
    vis2(train, test, predict, train_preds, test_preds, predict_preds, y_train, y_test)
    vis3(y_train, y_test, pred_graph)


# schedule.every().day.at("10:30").do(model)

# while True:
#         schedule.run_pending()
#         time.sleep(1)
