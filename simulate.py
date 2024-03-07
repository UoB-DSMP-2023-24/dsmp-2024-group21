import pandas as pd
import numpy as np
import data_processing
import config

def simulate_trades(lob_df):
    lob_df = lob_df.sort_values(by='timestamp')
    
    bids = lob_df[lob_df['Type'] == 'Bid'].copy()
    asks = lob_df[lob_df['Type'] == 'Ask'].copy()
    
    trades = []  

    for timestamp in sorted(set(lob_df['timestamp'].values)):
        current_bids = bids[bids['timestamp'] == timestamp]
        current_asks = asks[asks['timestamp'] == timestamp]
        
        # when highest bid match the lowest ask
        while not current_bids.empty and not current_asks.empty and current_bids['price'].max() >= current_asks['price'].min():
            highest_bid = current_bids[current_bids['price'] == current_bids['price'].max()].iloc[0]
            lowest_ask = current_asks[current_asks['price'] == current_asks['price'].min()].iloc[0]

            trade_price = lowest_ask['price']
            trade_quantity = min(highest_bid['quantity'], lowest_ask['quantity'])

            trades.append({'timestamp': timestamp, 'price': trade_price, 'quantity': trade_quantity})

            if highest_bid['quantity'] > trade_quantity:
                bids.loc[highest_bid.name, 'quantity'] -= trade_quantity
            else:
                bids = bids.drop(highest_bid.name)
            
            if lowest_ask['quantity'] > trade_quantity:
                asks.loc[lowest_ask.name, 'quantity'] -= trade_quantity
            else:
                asks = asks.drop(lowest_ask.name)

            # update current bids and asks
            current_bids = bids[bids['timestamp'] == timestamp]
            current_asks = asks[asks['timestamp'] == timestamp]

    trades_df = pd.DataFrame(trades)

    return trades_df


if __name__=='__main__':
    data_processing.preprocess_LOBs_data(config.LOBs_test_directory_path,config.LOBs_test_hdf5_path)
    lob_df=data_processing.load_LOBs_data_by_date(config.LOBs_test_hdf5_path,'2025-01-02')
    trades_df = simulate_trades(lob_df)
    print(trades_df)
