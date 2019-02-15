#import libraries
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ARGO Labs',
    'depends_on_past': False,
    'start_date': datetime(2019, 2, 12), #arbitrary for now
    'email': ['chris@argolabs.org'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'queue': 'bash_queue',
    'pool': 'backfill',
    'priority_weight': 10,
    'end_date': datetime(2019, 9, 30), #set it to end at the end of the snow year
}

#initiate dag
dag = DAG(
    'snowbot', default_args=default_args, schedule_interval=timedelta(days=1))
	
	#establish tasks
t1 = BashOperator(
    task_id='fetch_data',
    bash_command='fetchData.py',
    owner=default_args['owner'],
    retries=3,
    dag=dag)

t2 = BashOperator(
    task_id='process_data',
    bash_command='krigingscript.py',
    owner=default_args['owner'],
    retries=3,
    dag=dag)

t3 = BashOperator(
    task_id='run_model',
    bash_command='production_script.py',
    owner=default_args['owner'],
    retries=3,
    dag=dag)

t4 = BashOperator(
    task_id='twitter_post',
    bash_command='boto3.py',
    owner=default_args['owner'],
    retries=3,
    dag=dag)

t5 = BashOperator(
    task_id='daily_run',
    bash_command='dailyRuns.py',
    owner=default_args['owner'],
    retries=3,
    dag=dag)
	
#set dependencies
t1.set_downstream(t2)
t2.set_downstream(t3)
t3.set_downstream(t4)
t4.set_downstream(t5)