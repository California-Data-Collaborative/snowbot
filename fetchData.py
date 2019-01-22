import pandas as pd
import urllib
import datetime
import boto3
import os 
import re

bucket = 'cadc-snowbot'
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')

stationIDs = pd.read_csv('data/stationNames.csv').ID
first_date = '1980-07-02'
today = str(datetime.date.today())

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
    
def updateHistorical(sens=3, start_date=first_date, end_date=today):  
    st = '3LK'
    url = f'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={st}&SensorNums={sens}&dur_code=D&Start={start_date}&End={end_date}'
    
    df_raw = pd.read_csv(url)
    df_raw['station_id'] = st
    df = pd.DataFrame()
    
    for st in stationIDs[1:]:
        url = f'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={st}&SensorNums={sens}&dur_code=D&Start={start_date}&End={end_date}'
        tmp = pd.read_csv(url)
        
        df_raw = df_raw.append(tmp)
        
        tmp['date'] = [datetime.datetime.strptime(i[:8], '%Y%m%d').date() \
                       for i in tmp['DATE TIME']]
        tmp = tmp[['date', 'VALUE']]
        tmp.columns = ['date', st]
        tmp.set_index('date')

        df = tmp if df.shape[0]==0 else df.merge(tmp, how='outer')        
    
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
    
def getHistorical(sens='03', prefix='backup'):
    file = max([i for i in listAll() if re.match(r'%s.*\%s.csv'%(prefix, sens))])
    s3.resource.Object(bucket, f"data/{file}").download_file(filename=f'data/{file}')
    
def processRaw(filename):
    df = pd.read_csv(filename)
    for st in stationIDs:
        tmp = df[df.ID==st]
    
    
def dailyUpdate(sens=3):
    try:
        updateHistorical(sens)     # daily snow water content
    except:
        try:
            getHistorical(f'{sens:02}', 'backup')
        except:
            getHistorical(f'{sens:02}', 'raw')
            filename=max(os.listdir('data/raw*'))
            processRaw(filename)
            