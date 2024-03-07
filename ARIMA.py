from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pmdarima import auto_arima
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

def ARIMA(tapes_df):
    for date_df,date in tapes_df:
        base_date = datetime.strptime(date, '%Y-%m-%d')
        date_df['timestamp'] = date_df['timestamp'].apply(lambda x: base_date + timedelta(seconds=x))

        date_df.set_index('timestamp', inplace=True)

        resampled_series = date_df.resample('1S').mean()
        resampled_series.fillna(method='ffill', inplace=True)
        resampled_df = pd.DataFrame(resampled_series, columns=['price'])



        prices = resampled_df['price']

        auto_model = auto_arima(prices, start_p=0, start_q=0,
                                max_p=5, max_q=5, m=1,
                                d=None, seasonal=False,
                                D=0, trace=True,
                                error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True)

        print(auto_model.summary())

        order = auto_model.order
        model = sm.tsa.ARIMA(prices, order=order)
        model_fit = model.fit()

        # fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        # sm.graphics.tsa.plot_acf(prices, lags=40, ax=ax[0])
        # sm.graphics.tsa.plot_pacf(prices, lags=40, ax=ax[1])
        # plt.show()
        plot_data_with_differences(prices, f'Price and Differences for {date}')
    


dates=getDates(config.Tapes_directory_path)
tapes_df=[]
for date in dates:
    tapes_date_df = load_Tapes_data_by_date(config.Tapes_hdf5_path,date)
    tapes_df.append([tapes_date_df,date])

ARIMA(tapes_df)
