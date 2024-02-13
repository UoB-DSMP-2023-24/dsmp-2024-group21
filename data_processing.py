import ast
import pandas as pd
import os
import re
from datetime import datetime

# Directory path
directory_path = './test/data/LOBs'
dp='./'
# Function to parse bids and asks from the orders string
def parse_orders(orders_str):
    try:
    # use Python's ast (Abstract Syntax Trees) to evalute literal
        orders = ast.literal_eval(orders_str)
        bids=[]
        asks=[]
        for order_type, order_list in orders:
                if order_type == 'bid':
                    # Parse bids if present
                    bids = [(float(price), int(qty)) for price, qty in order_list]
                elif order_type == 'ask':
                    # Parse asks if present
                    asks = [(float(price), int(qty)) for price, qty in order_list]
        return bids, asks
    except ValueError as e:
        # Handle cases where the string cannot be evaluated
        print(f"Error parsing orders: {e}")
        return [], []
    
def preprocess_line(line):
    # Directly replace 'Exch0' with '"Exch0"' to ensure it's interpreted as a string
    # this is for ast.literal_eval. Because without "", Exch0 will trigger some errors in eval.
    line = line.replace('Exch0', '"Exch0"')
    return line

def load_data(directory_path):
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
                    row = {'Date': date_obj, 'Timestamp': timestamp, 'Exchange': exchange, 'Bids': bids, 'Asks': asks}
                    data_rows.append(row)   
    # Convert the list of rows into a DataFrame
    return pd.DataFrame(data_rows)

df=load_data(directory_path=directory_path)
print(df.head())