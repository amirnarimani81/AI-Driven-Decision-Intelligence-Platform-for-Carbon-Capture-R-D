import sqlite3
import pandas as pd

DB_PATH = "preprocessor_output.db"
TABLE_NAME = "process_data"

def run_query(query):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_data():
    return run_query(f"SELECT * FROM {TABLE_NAME}")