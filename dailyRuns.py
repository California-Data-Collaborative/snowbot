import fetchData as fd
# import datacollection as dc
import krigingscript as kg
import productionscript as ps
import boto3

def main():
#     bucket = 'cadc-snowbot'
#     s3_resource = boto3.resource('s3')
    print(':: Fetching daily data...')
    fd.updateHistorical()
    main_df = fd.getLastFile()   
    print(main_df.tail.index)
    today = fd.getToday()
#     dc.daily_collection(main_df)
    print(':: Processing daily data...')
#     kg.kriging_per_row(main_df.ix[-1:])
    kg.kriging_per_row(today)
    print(':: Generating predictions...')
    ps.production_job()
    print(':: Posting to Twitter...')
if __name__=="__main__":
    main()
