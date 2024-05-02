import pandas as pd
import numpy as np
from pyecharts import options as opts
import config
from data_processing import load_LOBs_data_by_date, load_Tapes_data_by_date, load_all_Tapes
from pyecharts.charts import Line, Scatter, Page





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
        # 添加价格和移动平均线的折线图
        chart.add_xaxis(tapes_df.index.astype(str).tolist())
        chart.add_yaxis("Price", tapes_df['price'].tolist(), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
        chart.add_yaxis("SMA10", tapes_df['SMA10'].tolist(), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
        chart.add_yaxis("SMA50", tapes_df['SMA50'].tolist(), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))

        # 处理买入卖出点，买入点用绿色标记，卖出点用红色标记
        buy_points = tapes_df[tapes_df['positions'] == 1]['price']
        sell_points = tapes_df[tapes_df['positions'] == -1]['price']
        
        # 创建散点图来表示买入和卖出点
        scatter = Scatter()
        scatter.add_xaxis(buy_points.index.astype(str).tolist())
        scatter.add_yaxis("Buy", buy_points.tolist(), label_opts=opts.LabelOpts(is_show=False), symbol_size=10, itemstyle_opts=opts.ItemStyleOpts(color="green"))
        
        scatter.add_xaxis(sell_points.index.astype(str).tolist())
        scatter.add_yaxis("Sell", sell_points.tolist(), label_opts=opts.LabelOpts(is_show=False), symbol_size=10, itemstyle_opts=opts.ItemStyleOpts(color="red"))

        # 将散点图覆盖在折线图上
        chart.overlap(scatter)

        # 应用全局配置
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

    page.render('portfolio_overview.html')


def backend(lob_hdf5, tapes_hdf5, date,interval='T'):
    lob_df = load_LOBs_data_by_date(lob_hdf5, date)
    tapes_df = load_Tapes_data_by_date(tapes_hdf5, date)
    
    # tapes_df=load_all_Tapes(config.AllTapes_hdf5_path)
    # lob_df=pd.read_hdf(config.AllLOBs_hdf5_path,'all_LOBs')
    lob_df.reset_index(drop=True, inplace=True)
    tapes_df.reset_index(drop=True, inplace=True)

    
    # print(lob_df.head(10))
    # print(tapes_df.head())
    
    initial_capital = 100000
    risk_per_trade = 0.04
    tapes_df['SMA10'] = tapes_df['price'].rolling(window=50).mean()
    tapes_df['SMA50'] = tapes_df['price'].rolling(window=100).mean()
    tapes_df['signal'] = np.where(tapes_df['SMA10'] > tapes_df['SMA50'], 1, 0)
    tapes_df['positions'] = tapes_df['signal'].diff().fillna(0)

    def buy_from_lob(amount, timestamp, lob_df):
        lob_df=lob_df[lob_df['rounded_second']==timestamp]
        lob_df=lob_df[lob_df['type']=='Ask']
        # print(lob_df.head(10))
        cost = 0
        volume = 0
        for _, row in lob_df.iterrows():
            trade_cost = min(amount,row['price']* row['quantity'])
            cost += trade_cost
            volume += min(trade_cost/row['price'], row['quantity'])
            amount -= trade_cost
            if amount <= 0:
                break
        return cost, volume
    
    def sell_from_lob(volume_to_sell, timestamp, lob_df):
        lob_df=lob_df[lob_df['rounded_second']==timestamp]
        lob_df=lob_df[lob_df['type']=='Bid']
        # print(lob_df.head(10))
        cost = 0
        volume=0
        for _, row in lob_df.iterrows():
            volume_iter= row['quantity']
            if volume+volume_iter>volume_to_sell:
                volume_iter=volume_to_sell-volume
                volume=volume_to_sell
            else:
                volume += volume_iter
            trade_cost = row['price']* volume_iter
            cost += trade_cost
            if volume==volume_to_sell:
                break
        return cost, volume

    # Initialize the portfolio DataFrame
    portfolio = pd.DataFrame(index=tapes_df.index)
    portfolio['positions'] = 0
    portfolio['cash'] = initial_capital

    for idx, row in tapes_df.iterrows():
        timestamp = row['timestamp']
        if idx == 0:
            portfolio.at[idx, 'positions'] = 0
        else:
            portfolio.at[idx, 'positions'] = portfolio.at[idx - 1, 'positions']
            portfolio.at[idx, 'cash'] = portfolio.at[idx - 1, 'cash']
        
  
        if row['positions'] != 0:
            current_cash = portfolio.loc[idx, 'cash']
            current_position = portfolio.at[idx, 'positions']
            risk_amount = current_cash * risk_per_trade
            
            if row['positions'] > 0:  
                cost, volume = buy_from_lob(risk_amount, timestamp, lob_df)
                portfolio.at[idx, 'positions'] += volume
                portfolio.at[idx, 'cash'] -= cost
            elif row['positions'] < 0 and current_position > 0:  
                volume_to_sell = current_position
                cost, volume = sell_from_lob(volume_to_sell, timestamp, lob_df)
                portfolio.at[idx, 'positions'] -= volume
                portfolio.at[idx, 'cash'] += cost
   

    # Calculate holdings and total values
    portfolio['holdings'] = portfolio['positions'] * tapes_df['price']
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()
    profit_rate=(portfolio["total"].iloc[-1]-initial_capital)/initial_capital
    print(f'profit rate is {profit_rate}')

    # print(portfolio.describe())
    # print(tapes_df[['price', 'SMA10', 'SMA50', 'signal', 'positions']].describe())
    render_charts(portfolio,tapes_df)
    


if __name__ == '__main__':
    backend(config.LOBs_hdf5_path, config.Tapes_hdf5_path, '2025-01-03') 

