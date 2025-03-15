"""This module contains utility functions for working with orders data."""

import pandas as pd
import sqlite3
import time
from datetime import datetime

def standardize_column_name(column_name):
    return column_name.strip().lower().replace(' ', '_')

def excel_to_sqlite_delta(excel_file: str, db_file: str, table_name: str, unique_key: str):
  """
  Reads an Excel file and updates an SQLite database with new records only.

  :param excel_file: Path to the Excel file.
  :param db_file: Path to the SQLite database file.
  :param table_name: Name of the table to store data in.
  :param unique_key: Column name that serves as a unique identifier.
  """
  try:
      # Load Excel file into DataFrame
      df = pd.read_excel(excel_file)
      df.columns = df.columns.map(standardize_column_name)

      # Connect to SQLite database
      with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()

        # Create table if not exists
        df.head(0).to_sql(table_name, conn, if_exists='append', index=False)

        # Fetch existing unique keys
        cursor.execute(f"SELECT {unique_key} FROM {table_name}")
        existing_keys = set(row[0] for row in cursor.fetchall())

        # Filter out already existing records
        new_data = df[~df[unique_key].isin(existing_keys)]

        if not new_data.empty:
            new_data.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"Added {len(new_data)} new records to {table_name}")
        else:
            print("No new records to add.")
  except Exception as e:
      import os
      print(os.getcwd())
      print(f"Error updating database: {e}")

def query_sqlite(db_file: str, query: str):
  """
  Queries an SQLite database and returns the results.
  """
  try:
      conn = sqlite3.connect(db_file)
      df = pd.read_sql_query(query, conn)
      return df
  except Exception as e:
      print(f"Error executing query: {e}")
      return None
  finally:
      conn.close()


