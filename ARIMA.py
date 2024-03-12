import LSTM

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def plot_time_series(df, column_name):
    ts = df[column_name]

    # 原始时间序列图
    plt.figure(figsize=(10, 6))
    plt.plot(ts)
    plt.title('Price Time Series')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

    # 一阶差分
    ts_diff1 = ts.diff(periods=1)
    plt.figure(figsize=(12, 6))
    plt.plot(ts_diff1, label='1st Order Differencing')
    plt.title('1st Order Differencing')
    plt.legend(loc='best')
    plt.show()

    # 二阶差分
    ts_diff2 = ts_diff1.diff(periods=1)
    plt.figure(figsize=(12, 6))
    plt.plot(ts_diff2, label='2nd Order Differencing')
    plt.title('2nd Order Differencing')
    plt.legend(loc='best')
    plt.show()

    # 自相关和偏自相关图
    fig, ax = plt.subplots(2, 1, figsize=(22, 20))
    plot_acf(ts, ax=ax[0])
    plot_pacf(ts, ax=ax[1])
    plt.show()
def prepare_series(df, timestamp_col, price_col):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
    df = df.set_index(timestamp_col)
    return df[price_col].dropna()

def split_data(series, train_ratio=0.99):
    split_point = int(len(series) * train_ratio)
    return series[:split_point], series[split_point:]

def train_arima(series, order):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def forecast(model, steps):
    return model.forecast(steps=steps)

def plot_forecast(actual, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(actual.index, actual, label='Actual')
    plt.plot(actual.index, forecast, label='Forecast')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.title('Forecast vs Actual Prices')
    plt.legend()
    plt.show()

def calculate_mse(actual, forecast):
    return mean_squared_error(actual, forecast)