
import pandas as pd
from pathlib import Path
from getpass import getpass
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "train_cleaned.csv"

# Read cleaned training data
df = pd.read_csv(DATA_PATH)

# Select and rename columns to match MySQL table
column_mapping = {
    "id": "passenger_id",
    "Gender": "gender",
    "Customer Type": "customer_type",
    "Age": "age",
    "Type of Travel": "travel_type",
    "Class": "travel_class",
    "Flight Distance": "flight_distance",
    "Departure Delay in Minutes": "departure_delay",
    "Arrival Delay in Minutes": "arrival_delay",
    "satisfaction": "satisfaction"
}

df = df[list(column_mapping.keys())].rename(columns=column_mapping)

print("=" * 55)
print("AIRLINE DATA - MYSQL IMPORT")
print("=" * 55)

print("\nRecords ready:", len(df))
print("Columns:", len(df.columns))

# Get password securely
password = getpass("Enter MySQL root password: ")

# Create database connection
connection_url = URL.create(
    drivername="mysql+pymysql",
    username="root",
    password=password,
    host="127.0.0.1",
    port=3306,
    database="airline_analytics"
)

engine = create_engine(connection_url)

try:
    with engine.begin() as connection:

        # Prevent accidental duplicate imports
        existing = connection.execute(
            text("SELECT COUNT(*) FROM passengers")
        ).scalar()

        if existing > 0:
            raise RuntimeError(
                f"Table already contains {existing} records. "
                "Import cancelled to prevent duplicates."
            )

        # Import in batches
        df.to_sql(
            name="passengers",
            con=connection,
            if_exists="append",
            index=False,
            chunksize=1000,
            method="multi"
        )

        total = connection.execute(
            text("SELECT COUNT(*) FROM passengers")
        ).scalar()

        print("\nMYSQL IMPORT SUCCESSFUL!")
        print("Total imported records:", total)

finally:
    engine.dispose()
    password = None

print("\nImport process completed.")
