from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import statsmodels.api as sm
import config
from data_processing import load_Tapes_data_by_date
from tools import getDates



def ARIMA(tapes_df):
    for date_df,date in tapes_df:
        base_date = datetime.strptime(date, '%Y-%m-%d')
        date_df['timestamp'] = date_df['timestamp'].apply(lambda x: base_date + timedelta(seconds=x))

        date_df.set_index('timestamp', inplace=True)

        resampled_series = date_df.resample('1S').mean()
        resampled_series.fillna(method='ffill', inplace=True)
        resampled_df = pd.DataFrame(resampled_series, columns=['price'])



        prices = resampled_df['price']

        model = sm.tsa.arima.ARIMA(prices, order=(1,1,1))

        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=5)
        print(forecast)

dates=getDates(config.Tapes_directory_path)
tapes_df=[]
for date in dates:
    tapes_date_df = load_Tapes_data_by_date(config.Tapes_hdf5_path,date)
    tapes_df.append([tapes_date_df,date])

ARIMA(tapes_df)
