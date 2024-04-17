import ast
import pandas as pd
import os
import re
import h5py
import numpy as np
from datetime import datetime
from tools import getDates
import config

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

def preprocess_all_Tapes_data(tapes_hdf5_path):
    dates=getDates(config.Tapes_directory_path)
    tapes_df=[]
    for date in dates:
        tapes_date_df = load_Tapes_data_by_date(tapes_hdf5_path,date)
        if tapes_date_df is not None:
            tapes_date_df['timestamp'] = pd.to_datetime(date) + pd.to_timedelta(tapes_date_df['timestamp'], unit='s')
            tapes_df.append(tapes_date_df)
            tapes_date_df.to_hdf(config.AllTapes_hdf5_path, key='all_tapes', mode='a', format='table', data_columns=True, append=True)
            print(f"Data for {date} appended to {config.AllTapes_hdf5_path}")
        else:
            print(f"No data available for {date}")
        print(tapes_df)
    return

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
            bids_dfs.append(pd.DataFrame(bids_data, columns=['timestamp', 'price', 'quantity']))
        if 'asks' in date_group:
            asks_data = date_group['asks'][:]
            asks_dfs.append(pd.DataFrame(asks_data, columns=['timestamp', 'price', 'quantity']))
        if bids_dfs:
            bids_df = pd.concat(bids_dfs)
            bids_df['Type'] = 'Bid'
        else:
            bids_df = pd.DataFrame(columns=['timestamp', 'price', 'quantity', 'type'])

        if asks_dfs:
            asks_df = pd.concat(asks_dfs)
            asks_df['Type'] = 'Ask'
        else:
            asks_df = pd.DataFrame(columns=['timestamp', 'price', 'quantity', 'type'])

        lob_df = pd.concat([bids_df, asks_df])

        # reset index
        lob_df.reset_index(drop=True, inplace=True)

        return lob_df

def preprocess_LOBs_data(directory_path, hdf5_path, batch_size=100000):
    file_pattern = re.compile(r'^UoB_Set01_(\d{4}-\d{2}-\d{2})LOBs\.txt$')
    processed_files_path = os.path.join(directory_path, 'processed.txt') 

    # 加载已处理文件列表
    try:
        with open(processed_files_path, 'r') as file:
            processed_files = set(file.read().splitlines())
    except FileNotFoundError:
        processed_files = set()

    # 循环处理每个符合条件的文件
    for filename in os.listdir(directory_path):
        match = file_pattern.match(filename)
        if match and filename not in processed_files:
            try:
                print(f"Processing file: {filename}")
                process_single_file(match, filename, directory_path, hdf5_path, batch_size)

                with open(processed_files_path, 'a') as pf:
                    pf.write(filename + '\n')
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def process_single_file(match, filename, directory_path, hdf5_path, batch_size):
    date_str = match.group(1)
    file_path = os.path.join(directory_path, filename)
    
    with h5py.File(hdf5_path, 'a') as hdf_file:
        date_group = hdf_file.require_group(date_str)
        for dataset_name in ["bids", "asks"]:
            if dataset_name not in date_group:
                date_group.create_dataset(dataset_name, shape=(0, 3), maxshape=(None, 3), dtype='f', chunks=True, compression="gzip")
                
        all_bids, all_asks = [], []
        with open(file_path, 'r') as file:
            for line in file:
                data_list = ast.literal_eval(preprocess_line(line.strip()))
                timestamp, exchange, orders = data_list
                bids, asks = parse_orders(str(orders), timestamp)
                all_bids.extend(bids)
                all_asks.extend(asks)

                if len(all_bids) >= batch_size:
                    append_to_dataset(date_group['bids'], np.array(all_bids, dtype='f'))
                    all_bids = []  
                if len(all_asks) >= batch_size:
                    append_to_dataset(date_group['asks'], np.array(all_asks, dtype='f'))
                    all_asks = []

            if all_bids:
                append_to_dataset(date_group['bids'], np.array(all_bids, dtype='f'))
            if all_asks:
                append_to_dataset(date_group['asks'], np.array(all_asks, dtype='f'))
    
def load_Tapes_data_by_date(hdf5_path, date):
    with h5py.File(hdf5_path, 'r') as hdf_file:
        date_group = hdf_file.get(date)
        if date_group is None:
            print(f"No data for date: {date}")
            return None
   
        if date in date_group:
            tapes_data = date_group[date][:]
            tapes_df=pd.DataFrame(tapes_data, columns=['timestamp', 'price', 'quantity'])
        return tapes_df

def list_datasets(hdf5_file):
    datasets = []

    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            datasets.append(name)

    hdf5_file.visititems(visitor_func)
    return datasets

def load_all_Tapes(hdf5_path):
    tapes_df=pd.read_hdf(hdf5_path, 'all_tapes')
    return tapes_df

if __name__=='__main__':
    print('1')
    
    # preprocess_all_Tapes_data(config.Tapes_hdf5_path)
    allTapes=load_all_Tapes(config.AllTapes_hdf5_path)
    # preprocess_LOBs_data(config.LOBs_directory_path,config.LOBs_hdf5_path)
    # preprocess_Tapes_data(config.Tapes_directory_path,config.Tapes_hdf5_path)   
    # lob_df=load_LOBs_data_by_date(config.LOBs_hdf5_path,'2025-01-02') 
    # tapes_df=load_Tapes_data_by_date(config.Tapes_hdf5_path,'2025-06-30') 
        
    # preprocess_LOBs_data(config.LOBs_test_directory_path,config.LOBs_test_hdf5_path)
    # preprocess_Tapes_data(config.Tapes_test_directory_path,config.Tapes_test_hdf5_path)   
    # lob_test_df=load_LOBs_data_by_date(config.LOBs_test_hdf5_path,'2025-01-02') 
    # tapes_test_df=load_Tapes_data_by_date(config.Tapes_test_hdf5_path,'2025-01-02') 
            
    # print(lob_df.describe())
    # print(lob_df.head())
    # print(tapes_df.describe())
    # print(tapes_df.head())
    
    # print(lob_test_df.describe())
    # print(lob_test_df.head())
    # print(tapes_test_df.describe())
    # print(tapes_test_df.head())
    # with h5py.File(config.LOBs_hdf5_path, 'r') as file:
    #     datasets = list_datasets(file)
    #     print("Datasets in the HDF5 file:")
    #     for dataset in datasets:
    #         print(dataset)

    #     print(f"Total number of datasets: {len(datasets)}")


