import pandas as pd
import numpy as np
from pyecharts import options as opts
import config
from data_processing import load_LOBs_data_by_date, load_Tapes_data_by_date, load_all_Tapes
from pyecharts.charts import Line, Scatter, Page
import json
from tools import getDates

def process_data(lob_hdf5,tapes_hdf5,date='2025-01-03', interval='T'):
    dates = getDates(config.LOBs_directory_path)
    # dates=['2025-01-02','2025-01-03']
    for date in dates:
        lob_date_df=load_LOBs_data_by_date(lob_hdf5,date)
        if lob_date_df is not None:
            ask_df=lob_date_df[lob_date_df['type']=='Ask']
            bid_df=lob_date_df[lob_date_df['type']=='Bid']
            
            grouped_ask = ask_df.groupby('rounded_second')
            new_ask_df = grouped_ask.apply(lambda x: pd.Series({
            'timestamp': x.name,
            'type': x['type'].iloc[0],
            'orders': list(x[['price', 'quantity']].to_dict('records'))
            })).reset_index(drop=True)
            
            grouped_bid = bid_df.groupby('rounded_second')
            new_bid_df = grouped_bid.apply(lambda x: pd.Series({
            'timestamp': x.name,
            'type': x['type'].iloc[0],
            'orders': list(x[['price', 'quantity']].to_dict('records'))
            })).reset_index(drop=True)
            
            lob_date_df=pd.concat([new_ask_df,new_bid_df],names=['timestamp','type', 'orders'])
            lob_date_df.reset_index(drop=True,inplace=True)
            lob_date_df.to_hdf('R_LOBs.h5', key=date, mode='a')
            # print(f"Data for {date} appended to lob_df")
        else:
            print(f"No data available for {date}")
    
 
    
    # tapes ohlc
    for date in dates:
        tapes_date_df=load_Tapes_data_by_date(tapes_hdf5,date)
        tapes_date_df.set_index('timestamp', inplace=True)
        ohlc = tapes_date_df['price'].resample(interval).ohlc()

        last_price = tapes_date_df['price'].resample(interval).last()
        last_quantity = tapes_date_df['quantity'].resample(interval).last()

        ohlc.dropna(inplace=True)


        last_df = pd.DataFrame({
            'timestamp': last_price.index,
            'price': last_price.values,
            'quantity': last_quantity.values
        })

        final_df = pd.merge(ohlc, last_df, on='timestamp')
        final_df.to_hdf('R_tapes.h5', key=date, mode='a')
        print(f"Data for {date} appended to tapes_df")
        
    
def load_data():
    dates = getDates(config.LOBs_directory_path)
    # dates=['2025-01-02','2025-01-03']
    lob_df = pd.DataFrame()
    tapes_df = pd.DataFrame()
    for date in dates:
        lob_date_df=pd.read_hdf('R_LOBs.h5',date)
        lob_df=pd.concat([lob_df,lob_date_df])
        print(lob_df.describe())
        tapes_date_df=pd.read_hdf('R_tapes.h5', key=date)
        tapes_df=pd.concat([tapes_df,tapes_date_df])
        print(tapes_df.describe())
        
    return lob_df,tapes_df

def render_charts(portfolio, tapes_df):
    page = Page()
    def create_chart(title, y_data):
        chart = Line()
        chart.add_xaxis(tapes_df.index.astype(str).tolist())
        chart.add_yaxis(
            series_name=title,
            y_axis=y_data,
            label_opts=opts.LabelOpts(is_show=False),  
            is_smooth=True
        )   
        chart.set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            datazoom_opts=[opts.DataZoomOpts()],
            yaxis_opts=opts.AxisOpts(is_scale=True),  
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
        return chart
    def create_price_chart(tapes_df):
        chart = Line()
        chart.add_xaxis(tapes_df.index.astype(str).tolist())
        chart.add_yaxis("Price", tapes_df['price'].tolist(), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
        chart.add_yaxis("SMA10", tapes_df['SMA10'].tolist(), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
        chart.add_yaxis("SMA50", tapes_df['SMA50'].tolist(), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
        chart.add_yaxis("stoploss", (tapes_df['price']-tapes_df['ATR']).tolist(), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
        chart.add_yaxis("takeprofit", (tapes_df['price']+tapes_df['ATR']).tolist(), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))

        buy_points = tapes_df[tapes_df['positions'] == 1]['price']
        sell_points = tapes_df[tapes_df['positions'] == -1]['price']

        scatter = Scatter()
        scatter.add_xaxis(buy_points.index.astype(str).tolist())
        scatter.add_yaxis("Buy", buy_points.tolist(), label_opts=opts.LabelOpts(is_show=False), symbol_size=10, itemstyle_opts=opts.ItemStyleOpts(color="green"))
        
        scatter.add_xaxis(sell_points.index.astype(str).tolist())
        scatter.add_yaxis("Sell", sell_points.tolist(), label_opts=opts.LabelOpts(is_show=False), symbol_size=10, itemstyle_opts=opts.ItemStyleOpts(color="red"))

        chart.overlap(scatter)

        chart.set_global_opts(
            title_opts=opts.TitleOpts(title="Price and Moving Averages"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            datazoom_opts=[opts.DataZoomOpts()],
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
            yaxis_opts=opts.AxisOpts(is_scale=True)
        )
        return chart

    line_cash = create_chart("Cash", portfolio['cash'].tolist())
    line_total = create_chart("Total", portfolio['total'].tolist())
    line_positions = create_chart("Positions", portfolio['positions'].tolist())
    line_holdings = create_chart("Holdings", portfolio['holdings'].tolist())
    line_price=create_price_chart(tapes_df)

    page.add(line_cash)
    page.add(line_total)
    page.add(line_positions)
    page.add(line_holdings)
    page.add(line_price)

    page.render('Rportfolio_overview.html')


def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return 0.8*atr

def strategy_entry_exit(row, atr, risk_per_trade, capital):
    entry_signal = row['positions']
    if entry_signal == 1:  # Long entry
        trade_risk = capital * risk_per_trade
        stop_loss = row['price'] - atr
        take_profit = row['price'] + 2 * atr
        return trade_risk, stop_loss, take_profit
    elif entry_signal == -1:  # Short entry
        trade_risk = capital * risk_per_trade
        stop_loss = row['price'] + atr
        take_profit = row['price'] - 2 * atr
        return trade_risk, stop_loss, take_profit
    return 0, 0, 0


def backend(lob_hdf5, tapes_hdf5, date,interval='T'):
    lob_df,tapes_df=load_data()
    lob_df.reset_index(drop=True, inplace=True)
    # tapes_df=load_all_Tapes(config.AllTapes_hdf5_path)
    # lob_df=pd.read_hdf(config.AllLOBs_hdf5_path,'all_LOBs')
    tapes_df.reset_index(drop=True, inplace=True)

   
    initial_capital = 10000
    risk_per_trade = 0.02
    tapes_df['SMA10'] = tapes_df['price'].rolling(window=50).mean()
    tapes_df['SMA50'] = tapes_df['price'].rolling(window=100).mean()
    tapes_df['ATR'] = calculate_atr(tapes_df)
    tapes_df['signal'] = np.where(tapes_df['SMA10'] > tapes_df['SMA50'], 1, 0)
    tapes_df['positions'] = tapes_df['signal'].diff().fillna(0)
    def buy_amount(amount, timestamp, lob_df):
        lob_df = lob_df[(lob_df['timestamp'] == timestamp) & (lob_df['type'] == 'Ask')]
        
        if not lob_df.empty and 'orders' in lob_df.iloc[0]:
            orders = lob_df.iloc[0]['orders']  
            cost = 0
            volume = 0
            for order in orders:
                if amount <= 0:
                    break
                trade_cost = min(amount, order['price'] * order['quantity'])
                cost += trade_cost
                volume += min(trade_cost / order['price'], order['quantity'])
                amount -= trade_cost
            return cost, volume
        
        return 0, 0

    def buy_volume(volume_to_sell, timestamp, lob_df):
        lob_df = lob_df[(lob_df['timestamp'] == timestamp) & (lob_df['type'] == 'Ask')]
        
        if not lob_df.empty and 'orders' in lob_df.iloc[0]:
            orders = lob_df.iloc[0]['orders']  
            cost = 0
            volume = 0
            
            for order in orders:
                if volume >= volume_to_sell:
                    break
                volume_iter = order['quantity']
                
                if volume + volume_iter > volume_to_sell:
                    volume_iter = volume_to_sell - volume
                    volume = volume_to_sell
                else:
                    volume += volume_iter
                
                trade_cost = order['price'] * volume_iter
                cost += trade_cost

                if volume >= volume_to_sell:
                    break

            return cost, volume
        
        return 0, 0
    
    def sell_amount(amount, timestamp, lob_df):
        lob_df = lob_df[(lob_df['timestamp'] == timestamp) & (lob_df['type'] == 'Bid')]
        
        if not lob_df.empty and 'orders' in lob_df.iloc[0]:
            orders = lob_df.iloc[0]['orders']  
            cost = 0
            volume = 0
            for order in orders:
                if amount <= 0:
                    break
                trade_cost = min(amount, order['price'] * order['quantity'])
                cost += trade_cost
                volume += min(trade_cost / order['price'], order['quantity'])
                amount -= trade_cost
            return cost, volume
        
        return 0, 0

    def sell_volume(volume_to_sell, timestamp, lob_df):
        lob_df = lob_df[(lob_df['timestamp'] == timestamp) & (lob_df['type'] == 'Bid')]
        
        if not lob_df.empty and 'orders' in lob_df.iloc[0]:
            orders = lob_df.iloc[0]['orders']  
            cost = 0
            volume = 0
            
            for order in orders:
                if volume >= volume_to_sell:
                    break
                volume_iter = order['quantity']
                
                if volume + volume_iter > volume_to_sell:
                    volume_iter = volume_to_sell - volume
                    volume = volume_to_sell
                else:
                    volume += volume_iter
                
                trade_cost = order['price'] * volume_iter
                cost += trade_cost

                if volume >= volume_to_sell:
                    break

            return cost, volume
        
        return 0, 0
    
    def long_watcher(stop_loss, take_profit, idx_begin, lob_df, portfolio):
        updated_portfolio = portfolio.copy()
        for idx, row in tapes_df.loc[idx_begin:].iterrows():
            updated_portfolio.at[idx, 'long'] = updated_portfolio.at[idx - 1, 'long']
            updated_portfolio.at[idx, 'cash'] = updated_portfolio.at[idx - 1, 'cash']
            current_long = updated_portfolio.at[idx, 'long']
            timestamp = row['timestamp']
            if row['price'] >= take_profit or row['price'] <= stop_loss:
                cost, volume = sell_volume(current_long, timestamp, lob_df)
                updated_portfolio.at[idx, 'long'] -= volume
                updated_portfolio.at[idx, 'cash'] += cost
                if updated_portfolio.at[idx, 'long'] <= 0:
                    break
        return updated_portfolio


    def short_watcher(stop_loss, take_profit,idx_begin,lob_df,portfolio):
        updated_portfolio = portfolio.copy()
        for idx, row in tapes_df.iterrows():
            if idx<=idx_begin:
                continue
            updated_portfolio.at[idx, 'short'] = updated_portfolio.at[idx - 1, 'short']
            updated_portfolio.at[idx, 'cash'] = updated_portfolio.at[idx - 1, 'cash']
            current_short=updated_portfolio.at[idx, 'short']
            timestamp = row['timestamp']
            if row['price']<=take_profit or row['price']>=stop_loss:
                if updated_portfolio.at[idx, 'short']<0:
                    cost, volume =buy_volume(-current_short,timestamp,lob_df)
                    updated_portfolio.at[idx, 'short'] += volume  
                    updated_portfolio.at[idx, 'cash'] -= cost
                else:
                    break
        return updated_portfolio

    # Initialize the portfolio DataFrame
    portfolio = pd.DataFrame(index=tapes_df.index)
    portfolio['positions'] = 0
    portfolio['long'] = 0
    portfolio['short'] = 0
    portfolio['cash'] = initial_capital

    for idx, row in tapes_df.iterrows():
        timestamp = row['timestamp']
        atr = row['ATR']
        if idx == 0:
            portfolio.at[idx, 'positions'] = 0
        else:
            portfolio.at[idx, 'positions'] = portfolio.at[idx - 1, 'positions']
            if portfolio.at[idx, 'cash'] == initial_capital:
                portfolio.at[idx, 'cash'] = portfolio.at[idx - 1, 'cash']
        current_cash = portfolio.loc[idx, 'cash']
        risk_amount, stop_loss, take_profit = strategy_entry_exit(row, atr, risk_per_trade, current_cash)


        # long
        if row['positions'] == 1:
            # buy risk amonut
            cost, volume = buy_amount(risk_amount, timestamp, lob_df)
            portfolio.at[idx, 'long'] += volume  
            portfolio.at[idx, 'positions'] += volume  
            portfolio.at[idx, 'cash'] -= cost
            # sell volume holding
            portfolio = long_watcher(stop_loss, take_profit, idx, lob_df, portfolio)
            
            
        # short
        elif row['positions'] == -1:
            cost, volume = sell_amount(risk_amount, timestamp, lob_df)
            portfolio.at[idx, 'positions'] = -volume  
            portfolio.at[idx, 'short'] -= volume  
            portfolio.at[idx, 'cash'] += cost
            portfolio =short_watcher(stop_loss,take_profit,idx,lob_df,portfolio)
            # print(1)
            

    # Calculate holdings and total values
    portfolio['holdings'] = portfolio['positions'] * tapes_df['price']
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()
    profit_rate=(portfolio["total"].iloc[-1]-initial_capital)/initial_capital
    print(f'profit rate is {profit_rate}')

    render_charts(portfolio,tapes_df)
    

if __name__ == '__main__':
    # process_data(config.LOBs_hdf5_path,config.Tapes_hdf5_path,'2025-01-03')
    backend(config.LOBs_hdf5_path, config.Tapes_hdf5_path, '2025-01-03',interval='T')
