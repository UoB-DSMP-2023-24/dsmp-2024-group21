import ast
import pandas as pd
import os
import re
import h5py
import numpy as np
from datetime import datetime

# Directory path
LOBs_directory_path = './test/data/LOBs'
Tapes_directory_path = './test/data/Tapes'


def parse_orders(orders_str):
    try:
        orders = ast.literal_eval(orders_str)
        bids = [(float(price), float(quantity))
                for price, quantity in orders[0][1]] if orders[0][1] else []
        asks = [(float(price), float(quantity))
                for price, quantity in orders[1][1]] if orders[1][1] else []
        return bids, asks
    except ValueError as e:
        print(f"Error parsing orders: {e}")
        return [], []


def preprocess_line(line):
    # Directly replace 'Exch0' with '"Exch0"' to ensure it's interpreted as a string
    # this is for ast.literal_eval. Because without "", Exch0 will trigger some errors in eval.
    line = line.replace('Exch0', '"Exch0"')
    return line


def load_LOBs_data(directory_path):
    # Use regex to select filename pattern and identify the date
    file_pattern = re.compile(r'^UoB_Set01_(\d{4}-\d{2}-\d{2})LOBs\.txt$')
    data_rows = []  # List to collect data rows

    for filename in os.listdir(directory_path):
        match = file_pattern.match(filename)
        if match:
            date_str = match.group(1)
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            file_path = os.path.join(directory_path, filename)

            with open(file_path, 'r') as file:
                for line in file:
                    # Directly evaluate the line as a Python list
                    data_list = ast.literal_eval(preprocess_line(line.strip()))
                    # Extract components directly from the list
                    timestamp, exchange, orders = data_list
                    bids, asks = parse_orders(str(orders))
                    # Prepare the row and add it to the list
                    # encode bid to 0 and ask to 1
                    for price, quantity in bids:
                        data_rows.append({'date': date_obj, 'timestamp': timestamp,
                                         'type': 'bid', 'price': price, 'quantity': quantity})
                    for price, quantity in asks:
                        data_rows.append({'date': date_obj, 'timestamp': timestamp,
                                         'type': 'ask', 'price': price, 'quantity': quantity})
    df = pd.DataFrame(data_rows, columns=[
                      'date', 'timestamp', 'type', 'price', 'quantity'])

    # Convert 'type' to 0 for bids and 1 for asks
    df['type'] = df['type'].map({'bid': 0, 'ask': 1}).astype(np.int8)
    # Convert 'price' and 'quantity' to suitable numeric types
    df['price'] = df['price'].astype(np.float32)
    df['quantity'] = df['quantity'].astype(np.int32)
    return df


def load_Tapes_data(directory_path):
    # Use regex to select filename pattern and identify the date
    file_pattern = re.compile(r'^UoB_Set01_(\d{4}-\d{2}-\d{2})tapes\.csv$')
    data_frames = []  # List to collect tapes data by date

    for filename in os.listdir(directory_path):
        match = file_pattern.match(filename)
        if match:
            date_str = match.group(1)
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path, names=[
                             'timestamp', 'price', 'quantity'])
            df['date'] = date_obj
            data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

def store_hdf5(dataframe, hdf5_path):
    dataframe.to_hdf(hdf5_path, key='lob', format='table',
                     data_columns=True, complib='blosc', complevel=9)


hdf5_path = 'data/hdf5/LOB.h5'

df_lob = load_LOBs_data(directory_path=LOBs_directory_path)
# df_tape=load_Tapes_data(directory_path=Tapes_directory_path)
store_hdf5(df_lob, hdf5_path)

# print("DataFrame Info:")
# df.info()
# print("\nDataFrame Data Types:")
# print(df.dtypes)

# print("\nStatistical Summary of Numeric Columns:")
# print(df.describe())
print(df_lob.head())
# print(df_tape.head())
