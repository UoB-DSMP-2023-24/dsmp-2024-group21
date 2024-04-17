from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pmdarima import ARIMA, auto_arima
import config
from data_processing import load_Tapes_data_by_date
from tools import getDates

def plot_data_with_differences(prices, title):
    plt.figure(figsize=(14, 7))
    
    # Plot original prices
    plt.subplot(3, 1, 1)
    plt.plot(prices, label='Original Prices')
    plt.title(title)
    plt.legend()
    
    # Plot 1st difference
    plt.subplot(3, 1, 2)
    plt.plot(prices.diff(), label='1st Order Difference')
    plt.title('1st Order Difference')
    plt.legend()

    # Plot 2nd difference
    plt.subplot(3, 1, 3)
    plt.plot(prices.diff().diff(), label='2nd Order Difference')
    plt.title('2nd Order Difference')
    plt.legend()

    plt.tight_layout()
    plt.show()

def process_data(date_df, date):
    base_date = datetime.strptime(date, '%Y-%m-%d')
    date_df['timestamp'] = date_df['timestamp'].apply(lambda x: base_date + timedelta(seconds=x))
    date_df.set_index('timestamp', inplace=True)
    resampled_series = date_df.resample('1S').mean()
    resampled_series.fillna(method='ffill', inplace=True)
    return resampled_series['price']

def ARIMA_forecast(prices, date):
    train_size = int(len(prices) * 0.8)
    train, test = prices.iloc[:train_size], prices.iloc[train_size:]
    
    model = sm.tsa.ARIMA(train, order=(0, 1, 5))
    model_fit = model.fit()
    
    # Forecast
    forecast_steps = len(test)
    forecast = model_fit.forecast(steps=forecast_steps)

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(prices.index, prices, label='Original Prices')
    plt.plot(test.index, forecast, label='Forecasted Prices', color='red')
    plt.title(f'Price and Forecast for {date}')
    plt.legend()
    plt.show()


def rolling_forecast_ARIMA(prices, order):
    train_size = int(len(prices) * 0.8)
    # Split the data into train and test
    train, test = prices[:train_size], prices[train_size:]
    
    # Fit the initial ARIMA model
    model = ARIMA(order=order)
    model.fit(train)
    
    # Lists to hold actual and predicted prices
    history = train.tolist()
    predictions = []
    predicted_index = []

    # Rolling forecast
    for t in range(len(test)):
        # Predict the next step
        yhat = model.predict(n_periods=1)
        predictions.append(yhat)
        predicted_index.append(test.index[t])

        # Update the model with the next actual value
        model.update(test.iloc[t])
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.plot(prices.index, prices, label='Historical Data')
    plt.plot(predicted_index, predictions, color='red', label='Rolling Forecast')
    plt.legend()
    plt.title('Historical Data and Rolling Forecast')
    plt.show()


dates=getDates(config.Tapes_directory_path)
tapes_df=[]
for date in dates:
    tapes_date_df = load_Tapes_data_by_date(config.Tapes_hdf5_path,date)
    tapes_df.append([tapes_date_df,date])



for date_df, date in tapes_df:
    prices = process_data(date_df.head(1000), date)  # Process your data
    # ARIMA_forecast(prices, date)  # Forecast and plot
    rolling_forecast_ARIMA(prices, order=(0, 1, 5))
    
