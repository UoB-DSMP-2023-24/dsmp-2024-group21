import ast
import pandas as pd
import os
import re
import h5py
import numpy as np
from datetime import datetime
import config
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

def parse_orders(orders_str,timestamp):
    try:
        orders = ast.literal_eval(orders_str)
        bids = [(float(timestamp),float(price), float(quantity))
                for price, quantity in orders[0][1]] if orders[0][1] else np.empty((0, 3), dtype='f')
        asks = [(float(timestamp),float(price), float(quantity))
                for price, quantity in orders[1][1]] if orders[1][1] else np.empty((0, 3), dtype='f')
        return bids, asks
    except ValueError as e:
        print(f"Error parsing orders: {e}")
        return [], []


def preprocess_line(line):
    # Directly replace 'Exch0' with '"Exch0"' to ensure it's interpreted as a string
    # this is for ast.literal_eval. Because without "", Exch0 will trigger some errors in eval.
    line = line.replace('Exch0', '"Exch0"')
    return line

def append_to_dataset(dataset, data_to_append):
    current_size = dataset.shape[0]
    additional_size = data_to_append.shape[0]
    dataset.resize(current_size + additional_size, axis=0)
    dataset[current_size:] = data_to_append

def preprocess_LOBs_data(directory_path, hdf5_path,batch_size=100000):
    file_pattern = re.compile(r'^UoB_Set01_(\d{4}-\d{2}-\d{2})LOBs\.txt$')
    processed_files_path = os.path.join(directory_path, 'processed.txt') 
    # load processed files list
    try:
        with open(processed_files_path, 'r') as file:
            processed_files = set(file.read().splitlines())
    except FileNotFoundError:
        processed_files = set()

    # 'a' parameter means open file or create a new one if it isn't exist
    with h5py.File(hdf5_path, 'a') as hdf_file:  
        for filename in os.listdir(directory_path):
            match = file_pattern.match(filename)
            if match and filename not in processed_files:
                print(f"Processing file: {filename}")
                date_str = match.group(1)
                file_path = os.path.join(directory_path, filename)
                date_group = hdf_file.require_group(date_str)
                for dataset_name in ["bids", "asks"]:
                    if dataset_name not in date_group:
                     date_group.create_dataset(dataset_name, shape=(0, 3), maxshape=(None, 3), dtype='f', chunks=True, compression="gzip")

                # accumulate bids and aska to save in batch
                all_bids, all_asks = [], []
                with open(file_path, 'r') as file:
                    for line in file:
                        data_list = ast.literal_eval(preprocess_line(line.strip()))
                        timestamp, exchange, orders = data_list
                        bids, asks = parse_orders(str(orders),timestamp)
                        all_bids.extend(bids)
                        all_asks.extend(asks)
                        if len(all_bids) >= batch_size:
                            append_to_dataset(date_group['bids'], np.array(all_bids, dtype='f'))
                            # reset all bids
                            all_bids = []  
                        if len(all_asks) >= batch_size:
                            append_to_dataset(date_group['asks'], np.array(all_asks, dtype='f'))
                            # reset all asks
                            all_asks = []  
                    # don't forget to process the rest
                    if all_bids:
                        append_to_dataset(date_group['bids'], np.array(all_bids, dtype='f'))
                    if all_asks:
                        append_to_dataset(date_group['asks'], np.array(all_asks, dtype='f'))

                    # update processed file list
                    with open(processed_files_path, 'a') as pf:
                        pf.write(filename + '\n')


def preprocess_Tapes_data(directory_path,hdf5_path):
    # Use regex to select filename pattern and identify the date
    file_pattern = re.compile(r'^UoB_Set01_(\d{4}-\d{2}-\d{2})tapes\.csv$')
    processed_files_path = os.path.join(directory_path, 'processed.txt') 
    # load processed files list
    try:
        with open(processed_files_path, 'r') as file:
            processed_files = set(file.read().splitlines())
    except FileNotFoundError:
        processed_files = set()
        
    data_frames = []  # List to collect tapes data by date
    with h5py.File(hdf5_path, 'a') as hdf_file: 
        for filename in os.listdir(directory_path):
            match = file_pattern.match(filename)
            if match and filename not in processed_files:
                date_str = match.group(1)
                # create group date
                date_group = hdf_file.require_group(date_str) 
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                file_path = os.path.join(directory_path, filename)
                df = pd.read_csv(file_path, names=[
                                'timestamp', 'price', 'quantity'])
                date_group.create_dataset(name=date_str,data=df.to_numpy())
                                # update processed file list
                with open(processed_files_path, 'a') as pf:
                        pf.write(filename + '\n')


def load_LOBs_data_by_date(hdf5_path, date):
    with h5py.File(hdf5_path, 'r') as hdf_file:
        date_group = hdf_file.get(date)
        if date_group is None:
            print(f"No data for date: {date}")
            return None

        bids_dfs = []
        asks_dfs = []

        if 'bids' in date_group:
            bids_data = date_group['bids'][:]
            bids_dfs.append(pd.DataFrame(bids_data, columns=['Timestamp', 'Price', 'Quantity']))

        if 'asks' in date_group:
            asks_data = date_group['asks'][:]
            asks_dfs.append(pd.DataFrame(asks_data, columns=['Timestamp', 'Price', 'Quantity']))

        if bids_dfs:
            bids_df = pd.concat(bids_dfs)
            bids_df['Type'] = 'Bid'
        else:
            bids_df = pd.DataFrame(columns=['Timestamp', 'Price', 'Quantity', 'Type'])

        if asks_dfs:
            asks_df = pd.concat(asks_dfs)
            asks_df['Type'] = 'Ask'
        else:
            asks_df = pd.DataFrame(columns=['Timestamp', 'Price', 'Quantity', 'Type'])

        lob_df = pd.concat([bids_df, asks_df])

        # reset index
        lob_df.reset_index(drop=True, inplace=True)

        return lob_df
    
def load_Tapes_data_by_date(hdf5_path, date):
    with h5py.File(hdf5_path, 'r') as hdf_file:
        date_group = hdf_file.get(date)
        if date_group is None:
            print(f"No data for date: {date}")
            return None
   
        if date in date_group:
            tapes_data = date_group[date][:]
            tapes_df=pd.DataFrame(tapes_data, columns=['Timestamp', 'Price', 'Quantity'])
        return tapes_df

import ARIMA





if __name__=='__main__':
    print('1')
    #preprocess_LOBs_data(config.LOBs_directory_path,config.LOBs_hdf5_path)
    #preprocess_Tapes_data(config.Tapes_directory_path,config.Tapes_hdf5_path)   
    lob_df=load_LOBs_data_by_date(config.LOBs_hdf5_path,'2025-01-02') 
    tapes_df=load_Tapes_data_by_date(config.Tapes_hdf5_path,'2025-01-02') 
    print(tapes_df.head(10000000))
    price_series = ARIMA.prepare_series(tapes_df, 'Timestamp', 'Price')
    train_series, test_series = ARIMA.split_data(price_series)
    model_fit = ARIMA.train_arima(train_series, (2, 0, 2))
    forecast_values = ARIMA.forecast(model_fit, len(test_series))
    #print(forecast_values)
    ARIMA.plot_forecast(test_series, forecast_values)
    #mse = calculate_mse(test_series, forecast_values)
    #print(f'Mean Squared Error: {mse}')
    #LSTM.process_and_predict(tapes_df)