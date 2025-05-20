from datetime import datetime, timedelta
import os
import logging
import math

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from airflow import DAG
from airflow.operators.python import PythonOperator

# Set up logging
task_log = logging.getLogger("airflow.task")

# Default args for all tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# DAG definition
with DAG(
    dag_id='flight_price_pipeline',
    default_args=default_args,
    description='ETL, train, evaluate: flight-price regression',
    schedule_interval='@daily',
    start_date=datetime(2025, 5, 18),
    catchup=False,
    tags=['flight_price'],
) as dag:

    def extract():
        task_log.info("Starting extract task")
        raw_dir = '/app/data'
        os.makedirs(raw_dir, exist_ok=True)
        df = pd.read_csv('/app/flights.csv')
        raw_path = os.path.join(raw_dir, 'raw_flights.parquet')
        df.to_parquet(raw_path, index=False)
        task_log.info(f"Extracted {len(df)} rows to {raw_path}")
        return raw_path

    def transform(**kwargs):
        task_log.info("Starting transform task")
        raw_path = kwargs['ti'].xcom_pull(task_ids='extract')
        df = pd.read_parquet(raw_path)
        task_log.info(f"Rows before clean: {len(df)}")
        df['time'] = df['time'].astype(float)
        df['distance'] = df['distance'].astype(float)
        clean_dir = '/app/data'
        clean_path = os.path.join(clean_dir, 'clean_flights.parquet')
        df.to_parquet(clean_path, index=False)
        task_log.info(f"Transform completed; wrote {len(df)} rows to {clean_path}")
        return clean_path

    def train(**kwargs):
        task_log.info("Starting train task")
        clean_path = kwargs['ti'].xcom_pull(task_ids='transform')
        df = pd.read_parquet(clean_path)
        X = df[['time', 'distance']]
        y = df['price']

        model = LinearRegression(positive=True)
        model.fit(X, y)

        models_dir = '/app/models'
        os.makedirs(models_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d')
        model_filename = f'flight_price_model_{timestamp}.joblib'
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        task_log.info(f"Train completed; model saved to {model_path}")
        return model_path

    def evaluate(**kwargs):
        task_log.info("Starting evaluate task")
        clean_path = kwargs['ti'].xcom_pull(task_ids='transform')
        model_path = kwargs['ti'].xcom_pull(task_ids='train')

        df = pd.read_parquet(clean_path)
        X = df[['time', 'distance']]
        y = df['price']

        task_log.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        preds = model.predict(X)

        # Calculate RMSE and R2 manually for backward compatibility
        mse = mean_squared_error(y, preds)
        rmse = math.sqrt(mse)
        r2 = r2_score(y, preds)

        metrics_dir = '/app/models'
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, f'metrics_{datetime.now().strftime("%Y%m%d")}.txt')

        try:
            with open(metrics_path, 'w') as f:
                f.write(f'RMSE: {rmse:.2f}\nR2: {r2:.3f}')
            task_log.info(f"Evaluate completed; metrics written to {metrics_path}")
        except Exception as e:
            task_log.error(f"Failed to write metrics file: {str(e)}")
            raise

        return metrics_path

    # Define tasks
    t1 = PythonOperator(task_id='extract', python_callable=extract)
    t2 = PythonOperator(task_id='transform', python_callable=transform)
    t3 = PythonOperator(task_id='train', python_callable=train)
    t4 = PythonOperator(task_id='evaluate', python_callable=evaluate)

    # Task dependencies
    t1 >> t2 >> t3 >> t4
