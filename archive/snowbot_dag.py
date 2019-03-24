#import libraries

from airflow.operators.python_operator import PythonOperator
from airflow.operators.check_operator import CheckOperator
from airflow.models import Variable
from airflow import DAG

import snowbot as sb

from datetime import datetime, timedelta


dag = DAG(
	dag_id = 'snowbot',
	schedule_interval='@daily',
	start_date=datetime(2019, 2, 26)
	)

#initialize tasks	

fetch_data = PythonOperator(
	task_id="fetch_data",
	python_callable=sb.fetchData,
	dag=dag
	)
	
process_data = PythonOperator(
	task_id="process_data",
	python_callable=sb.krigingscript,
	dag=dag
	)
	
run_model = PythonOperator(
	task_id="run_model",
	python_callable=sb.product_script,
	dag=dag
	)
	
twitter_post = PythonOperator(
	task_id="twitter_post",
	python_callable=sb.boto3,
	dag=dag
	)
	
#set dependencies
fetch_data.set_downstream(process_data)
process_data.set_downstream(run_model)
run_model.set_downstream(twitter_post)
