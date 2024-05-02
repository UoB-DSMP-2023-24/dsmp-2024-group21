import pandas as pd
import numpy as np
from pyecharts import options as opts
import config
from data_processing import load_LOBs_data_by_date, load_Tapes_data_by_date, load_all_Tapes
from pyecharts.charts import Line, Scatter, Page
from tools import getDates
from talib import abstract
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt


    
def load_data():
    # dates=getDates(config.LOBs_directory_path)
    dates=['2025-01-02','2025-01-03']
    lob_df = pd.DataFrame()
    tapes_df = pd.DataFrame()
    for date in dates:
        # lob_date_df=pd.read_hdf('R_LOBs.h5',date)
        # lob_df=pd.concat([lob_df,lob_date_df])
        # print(lob_df.describe())
        tapes_date_df=pd.read_hdf('R_tapes.h5', key=date)
        tapes_df=pd.concat([tapes_df,tapes_date_df])
        print(tapes_df.describe())
        
    return lob_df,tapes_df

def render_charts(portfolio, tapes_df,test_data,signals):
    page = Page()
    from pyecharts.charts import Line, Scatter, Grid

    def create_predict_chart(tapes_df,test_data):
        line_chart = Line()
        line_chart.add_xaxis(tapes_df.index.tolist())
        line_chart.add_yaxis("Actual ", tapes_df['close'].tolist(), is_smooth=True)
        line_chart.add_yaxis("Predictions", test_data['predictions'].tolist(), is_smooth=True, linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.5))

        line_chart.set_global_opts(
            title_opts=opts.TitleOpts(title="Price and Moving Averages"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            datazoom_opts=[opts.DataZoomOpts()],
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
            yaxis_opts=opts.AxisOpts(is_scale=True)
        )
        return line_chart
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
        chart.add_yaxis("ATR", tapes_df['ATR'].tolist(), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
        chart.add_yaxis("MACD", tapes_df['MACD'].tolist(), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
        chart.add_yaxis("RSI", tapes_df['RSI'].tolist(), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))

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

    # line_cash = create_chart("Cash", portfolio['cash'].tolist())
    # line_macd = create_chart("MACD_signal", tapes_df['MACD_signal'].tolist())
    # line_price=create_price_chart(tapes_df)
    line_predict=create_predict_chart(tapes_df,test_data, signals)
    page.add(line_predict)

    # page.add(line_cash)
    # page.add(line_macd)
    # page.add(line_price)

    page.render('LSTM.html')


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


def generate_signals(predictions, threshold=0.5):
    signals = (predictions > threshold).astype(int)
    return signals

def plot_predictions(test_data, predictions, signals):
    plt.figure(figsize=(15, 7))
    plt.plot(test_data.index, test_data['close'], label='Actual Close Price', color='blue', alpha=0.6)
    plt.plot(test_data.index, predictions, label='Predicted Close Probability', color='red', alpha=0.6)

    buy_signals = test_data[signals == 1]
    plt.scatter(buy_signals.index, buy_signals['close'], label='Buy Signal', marker='^', color='green', alpha=1)

    sell_signals = test_data[signals == 0]
    plt.scatter(sell_signals.index, sell_signals['close'], label='Sell Signal', marker='v', color='red', alpha=1)

    plt.title('Comparison of Predictions and Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def LSTM_strategy():
    lob_df, tapes_df = load_data()
    tapes_df.reset_index(drop=True, inplace=True)
    
    tapes_df['SMA10'] = tapes_df['close'].rolling(window=10).mean()
    tapes_df['SMA50'] = tapes_df['close'].rolling(window=50).mean()
    tapes_df['ATR'] = calculate_atr(tapes_df)
    tapes_df['RSI'] = abstract.RSI(tapes_df['close'], timeperiod=14)
    tapes_df['MACD'], tapes_df['MACD_signal'], _ = abstract.MACD(tapes_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    tapes_df.dropna(inplace=True)   
    original_tapes_df = tapes_df.copy()

    feature_columns = ['SMA10', 'SMA50', 'ATR', 'RSI', 'MACD', 'MACD_signal', 'close']
    X = tapes_df[feature_columns]
    y = tapes_df['close']  

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    split_ratio = 0.8
    split = int(len(X_scaled) * split_ratio)
    X_train, y_train = X_scaled[:split], y_scaled[:split]
    X_test, y_test = X_scaled[split:], y_scaled[split:]

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='linear'))  # 输出层改为线性激活
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # 使用均方误差作为损失函数

    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    predictions = model.predict(X_test).flatten()
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()  # 反标准化预测结果
    test_data = original_tapes_df.iloc[split:]
    test_data['predictions'] = predictions

    line_chart = Line()
    line_chart.add_xaxis(test_data.index.astype(str).tolist())
    line_chart.add_yaxis("Actual ", tapes_df['close'].tolist(), is_smooth=True)
    line_chart.add_yaxis("Predictions", test_data['predictions'].tolist(), is_smooth=True, 
                         label_opts=opts.LabelOpts(is_show=False),
                         linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.5))

    line_chart.set_global_opts(
        title_opts=opts.TitleOpts(title="Price and Moving Averages"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        datazoom_opts=[opts.DataZoomOpts()],
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        yaxis_opts=opts.AxisOpts(is_scale=True)
    )
    line_chart.render('LSTM.html')    

    return test_data

# def LSTM_strategy():
#     lob_df, tapes_df = load_data()
#     tapes_df.reset_index(drop=True, inplace=True)
    
#     tapes_df['SMA10'] = tapes_df['close'].rolling(window=10).mean()
#     tapes_df['SMA50'] = tapes_df['close'].rolling(window=50).mean()
#     tapes_df['ATR'] = calculate_atr(tapes_df)
#     tapes_df['RSI'] = abstract.RSI(tapes_df['close'], timeperiod=14)
#     tapes_df['MACD'], tapes_df['MACD_signal'], _ = abstract.MACD(tapes_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
#     tapes_df.dropna(inplace=True)   
#     original_tapes_df = tapes_df.copy()

#     split_ratio = 0.8
#     split = int(len(tapes_df) * split_ratio)

    
#     feature_columns = ['SMA10', 'SMA50', 'ATR', 'RSI', 'MACD', 'MACD_signal', 'close']
#     X = tapes_df[feature_columns]
#     y = (tapes_df['SMA10'] > tapes_df['SMA50']).astype(int)

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     y = y.values

#     X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

#     X_train, y_train = X_scaled[:split], y[:split]
#     X_test, y_test = X_scaled[split:], y[split:]
#     assert not np.isnan(X_train).any(), "X_train contains NaN"
#     assert not np.isnan(y_train).any(), "y_train contains NaN"

#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=50, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=1, activation='linear')) 
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     # model.add(Dense(units=1, activation='sigmoid'))  
#     # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#     history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

#     performance = model.evaluate(X_test, y_test, verbose=0)
#     print('Test Loss and Accuracy:', performance)
    
#     predictions = model.predict(X_test).flatten()
#     test_data = original_tapes_df.iloc[split:]  
#     test_data['predictions'] = predictions

#     signals = generate_signals(predictions)
#     prices = test_data['close']

#     portfolio = backtest(signals, prices)
#     render_charts(portfolio, tapes_df,test_data,signals)
#     return 



def backtest(signals, prices, initial_capital=10000, transaction_cost=0.001):
    # 确保 signals 为 DataFrame 并设置正确的索引
    signals_df = pd.DataFrame(signals, index=prices.index, columns=['signals'])
    
    portfolio = pd.DataFrame(index=signals_df.index)
    portfolio['holdings'] = signals_df['signals'] * prices  
    portfolio['cash'] = initial_capital - (signals_df['signals'].iloc[0] * prices.iloc[0])

    for i in range(1, len(signals_df)):
        position_changes = signals_df['signals'].iloc[i] - signals_df['signals'].iloc[i - 1]
        portfolio['cash'].iloc[i] = portfolio['cash'].iloc[i - 1] - position_changes * prices.iloc[i] * (1 + transaction_cost)
    
    portfolio['total'] = portfolio['holdings'] + portfolio['cash']
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod() - 1
    final_value = portfolio['total'].iloc[-1]
    return_rate = (final_value - initial_capital) / initial_capital
    # print(f'return rate is {return_rate}')

    return portfolio






def backend(interval='T'):
    lob_df,tapes_df=load_data()
    lob_df.reset_index(drop=True, inplace=True)
    tapes_df.reset_index(drop=True, inplace=True)

   
    initial_capital = 10000
    risk_per_trade = 0.02
    

    tapes_df['SMA10'] = tapes_df['close'].rolling(window=10).mean()
    tapes_df['SMA50'] = tapes_df['close'].rolling(window=50).mean()
    tapes_df['ATR'] = calculate_atr(tapes_df)
    tapes_df['signal'] = np.where(tapes_df['SMA10'] > tapes_df['SMA50'], 1, 0)
    tapes_df['positions'] = tapes_df['signal'].diff().fillna(0)
    tapes_df['RSI'] = abstract.RSI(tapes_df['close'], timeperiod=14)
    tapes_df['MACD'], tapes_df['MACD_signal'], _ = abstract.MACD(tapes_df['close'])

   
    
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
            print(1)
            

    # Calculate holdings and total values
    portfolio['holdings'] = portfolio['positions'] * tapes_df['price']
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()
    profit_rate=(portfolio["total"].iloc[-1]-initial_capital)/initial_capital
    # print(f'profit rate is {profit_rate}')

    render_charts(portfolio,tapes_df)
    

if __name__ == '__main__':
    # backend()
    LSTM_strategy()
    
    # lob_df,tapes_df=load_data()
