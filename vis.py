

import pandas as pd
import plotly.graph_objects as go
import h5py
import config
from data_processing import load_LOBs_data_by_date



def update_top_orders(top_orders, new_order, n, order_type):
    """
    Update the top orders list with a new order and maintain its length as n.
    top_orders: DataFrame of current top orders (asks or bids).
    new_order: Series or dict with 'price' and other order details.
    n: int, number of top orders to maintain.
    order_type: 'ask' or 'bid', type of the order to update.
    """
    is_ask = (order_type == 'Ask')
    if is_ask:
        condition = (new_order['price'] < top_orders['price'].max()) or len(top_orders) < n
        ascending = True
    else:
        condition = (new_order['price'] > top_orders['price'].min()) or len(top_orders) < n
        ascending = False

    if condition:
        # Add the new order
        top_orders = pd.concat([top_orders, pd.DataFrame([new_order])])
        # Sort and maintain top n orders
        top_orders = top_orders.sort_values(by='price', ascending=ascending).head(n)

    return top_orders






lob_df=load_LOBs_data_by_date(config.LOBs_test_hdf5_path,'2025-01-03')
print(lob_df.head())
asks=lob_df[lob_df['Type']=='Ask']
print(f"asks: {asks}")
bids=lob_df[lob_df['Type']=='Bid']
print(f"bids: {bids}")

n = 5

def depth_chart():
    # Initialize figure
    fig = go.Figure()

    # Initial top n asks and bids
    top_asks=pd.DataFrame(columns=['timestamp','price','quantity','Type'])
    top_bids=pd.DataFrame(columns=['timestamp','price','quantity','Type'])

    # Initial Plot
    fig.add_trace(go.Scatter(x=top_bids['price'], y=top_bids['quantity'], mode='lines', name='Bids', fill='tozeroy'))
    fig.add_trace(go.Scatter(x=top_asks['price'], y=top_asks['quantity'], mode='lines', name='Asks', fill='tozeroy'))

    # Animation Frames
    frames = []
    timestamps = lob_df['timestamp'].unique()
    for ts in timestamps:
        df_update = lob_df[lob_df['timestamp'] == ts]
        new_asks=asks[asks['timestamp']==ts]
        new_bids=bids[bids['timestamp']==ts]
        if not new_asks.empty:
            top_asks= update_top_orders(top_asks,new_asks, n,'Ask')
        if not new_bids.empty:
            top_bids= update_top_orders(top_bids,new_bids,n,'Bid')
        frame = go.Frame(data=[
            go.Scatter(x=top_bids['price'], y=top_bids['quantity']),
            go.Scatter(x=top_asks['price'], y=top_asks['quantity'])
        ], name=str(ts))
        frames.append(frame)

    # Add frames to the figure
    fig.frames = frames

    # Add a slider to control the animation
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                        method="animate",
                        args=[None])])]
    )

    # Show the figure
    fig.show()
