import pandas as pd
import sqlite3
from pathlib import Path

DATA_DIR = Path('../data')
CSV_FILE = DATA_DIR / 'data.csv'
DB_FILE = DATA_DIR / 'project.db'
TABLE_NAME = 'Houses'

def migrate_csv_to_sql():
    """
    A one-time script to read data from the CSV, clean it,
    and load it into a new SQLite database.
    """
    try:
        df = pd.read_csv(CSV_FILE)
        df = df[df['price'] > 0]
        print(f"Pandas has {len(df)} rows ready to insert.")

        DATA_DIR.mkdir(exist_ok=True)
        conn = sqlite3.connect(DB_FILE)

        df.to_sql(
            name=TABLE_NAME,
            con=conn,
            if_exists='replace',
            index=False
        )

        print(f"Successfully migrated all {len(df)} rows to {DB_FILE}")

    except FileNotFoundError:
        print(f"Error: The file {CSV_FILE} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    migrate_csv_to_sql()