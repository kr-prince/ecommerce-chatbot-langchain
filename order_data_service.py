"""This is the main file for the order data service. It loads the order data
from a given source and stores it in the database every hour."""

from utils.orders import excel_to_sqlite_delta
from utils.configs import (
  ORDERS_EXCEL_PATH,
  DB_PATH,
  ORDERS_TABLE_NAME,
  UNIQUE_ID_COLUMN,
  SLEEP_TIME
)
import time
from datetime import datetime

def schedule_updates(excel_file: str, db_file: str, table_name: str, unique_key: str):
  """
  Runs the update process once and subsequently every hour, continuously.
  """
  while True:
      try:
          time_now = datetime.now()
          print(f"Running update at {time_now.strftime('%Y-%m-%d %H:%M:%S')}")
          excel_to_sqlite_delta(excel_file, db_file, table_name, unique_key)
      except Exception as e:
          print(f"Error in scheduled update: {e}")
      finally:
          time.sleep(SLEEP_TIME)

if __name__ == "__main__":
  schedule_updates(ORDERS_EXCEL_PATH, DB_PATH, ORDERS_TABLE_NAME, UNIQUE_ID_COLUMN)