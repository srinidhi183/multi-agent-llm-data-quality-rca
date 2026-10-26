
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def validate_olist():
    print("OLIST PIPELINE VALIDATION")
    print("Bronze -> Silver -> Gold")
    print("Airflow task executed successfully")


def validate_dataco():
    print("DATACO PIPELINE VALIDATION")
    print("Bronze -> Silver -> Gold")
    print("Airflow task executed successfully")


def validate_nyc_taxi():
    print("NYC TAXI PIPELINE VALIDATION")
    print("Bronze -> Silver -> Gold")
    print("Airflow task executed successfully")


with DAG(
    dag_id="thesis_pipeline_validation",
    description="Initial Airflow orchestration test",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    tags=["master-thesis", "validation"],
) as dag:

    olist = PythonOperator(
        task_id="validate_olist",
        python_callable=validate_olist,
    )

    dataco = PythonOperator(
        task_id="validate_dataco",
        python_callable=validate_dataco,
    )

    nyc_taxi = PythonOperator(
        task_id="validate_nyc_taxi",
        python_callable=validate_nyc_taxi,
    )

    olist >> dataco >> nyc_taxi
