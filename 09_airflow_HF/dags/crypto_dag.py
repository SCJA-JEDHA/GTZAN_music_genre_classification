from airflow import DAG
from datetime import datetime
import ccxt
import logging
import json
import os
import pandas as pd
from datetime import datetime


from airflow import DAG
from airflow.operators.python import PythonOperator



def _fetch_btc_infos(ti):
    # Fetch datas with ccxt
    hitbtc = ccxt.hitbtc()
    ticker = hitbtc.fetch_ticker("BTC/USDT")
    # Create a filename using the current timestamp as name
    now = datetime.now().timestamp()
    filename = f"{now}.json"
    # Dump the datas inside the JSON file
    with open(f"./data/{filename}", "w") as f:
        json.dump(ticker, f)
    # Logging is cool!
    logging.info(f"BTC dumped into {filename} 👍")
    ti.xcom_push(key="filename",value=filename)

def _transform_data(ti):
    filename = ti.xcom_pull(task_ids="fetch_btc_infos", key="filename")
    # Open the file
    f = open(f"./data/{filename}")
    data = json.load(f)
    f.close()
    # Keep only the keys we want
    data_process = {
        "value": data["symbol"],
        "datetime": data["datetime"],
        "price": data["info"]["ask"]
    }
    # Create a DataFrame
    df = pd.DataFrame(data_process, index=[0])
    # Append or create the CSV file
    if os.path.exists("./data/btc_history.csv"):
        df.to_csv("./data/btc_history.csv", mode="a", header=False, index=False)
    else:
        df.to_csv("./data/btc_history.csv", header=True, index=False)
    # Logging is sooo cool!
    logging.info("Data saved in btc_history.csv 🥳")


with DAG("crypto_dag", start_date=datetime(2026, 6, 7), schedule_interval="@hourly", catchup=False) as dag:
    
    fetch_btc_infos = PythonOperator(task_id="fetch_btc_infos", python_callable=_fetch_btc_infos)

    transform_data = PythonOperator(
        task_id = "transform_data",python_callable = _transform_data
    )

    fetch_btc_infos >> transform_data
    
