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
    url = f'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={st}&SensorNums={sens}&dur_code=D&Start={start_date}&End={end_date}'
    
    df_raw = pd.read_csv(url)
    df_raw['station_id'] = st
    df = pd.DataFrame()
    
    for st in stationIDs[1:]:
        url = f'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={st}&SensorNums={sens}&dur_code=D&Start={start_date}&End={end_date}'
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
    
    return(df, df_raw)
    
def backupData(df, df_raw, sens):
    fname = f"{today.replace('-','')}_{sens:02}"
    fnameBackup = f"backup_{fname}.csv"
    fnameRaw = f"raw_{fname}.csv"
    
    df.to_csv(f"data/{fnameBackup}")
    df_raw.to_csv(f"data/{fnameRaw}")
    
    s3_resource.Object(bucket, fnameBackup).upload_file(Filename=f"data/{fnameBackup}")
    s3_resource.Object(bucket, fnameRaw).upload_file(Filename=f"data/{fnameRaw}")
    
    if os.path.exists(f"data/{fnameBackup}"):
        os.remove(f"data/{fnameBackup}")
    if os.path.exists(f"data/{fnameRaw}"):
        os.remove(f"data/{fnameRaw}")
    
def getLastFile(sens='03', prefix='backup', download=False):
    r = re.compile("%s.*%s.csv"%(prefix, sens))
    f = [i for i in listAll() if r.match(i)]
    file = max(f)
    if download:
        s3_resource.Object(bucket, file).download_file(Filename=f'data/{file}')
    else:
        obj = s3_client.get_object(Bucket=bucket, Key=file)#.download_file(Filename=f'data/{file}')
        return(pd.read_csv(io.BytesIO(obj['Body'].read())))
    


def getUrl(st, sens, start, end):
    return(f'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={st}&SensorNums={sens}&dur_code=D&Start={today}&End={today}')

def getToday(start_date = yesterday):
    df, df_raw = updateHistorical(3, yesterday, yesterday)
    return(df)

def dailyUpdate(sens=3):
    try:
        updateHistorical(sens)     # daily snow water content
    except:
        try:
            getLastFile(f'{sens:02}', 'backup')
        except:
            getLastFile(f'{sens:02}', 'raw')
            filename=max(os.listdir('data/raw*'))
            processRaw(filename)
            