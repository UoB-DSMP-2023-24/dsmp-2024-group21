# LSTM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series)):
        end_ix = i + n_steps
        if end_ix > len(series)-1:
            break
        seq_x, seq_y = series[i:end_ix], series[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def process_and_predict(tapes_df, price_col='Price', n_steps=3, units=50, learning_rate=0.01, train_ratio=0.8):
    scaler = MinMaxScaler(feature_range=(0, 1))
    series = tapes_df[price_col]
    scaled_series = scaler.fit_transform(series.values.reshape(-1, 1))

    # 分割训练集和测试集
    split_idx = int(len(scaled_series) * train_ratio)
    train_series = scaled_series[:split_idx]
    test_series = scaled_series[split_idx:]

    X_train, y_train = prepare_data(train_series, n_steps)
    X_test, y_test = prepare_data(test_series, n_steps)

    n_features = 1
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

    # 构建 LSTM 模型
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    # 拟合模型
    model.fit(X_train, y_train, epochs=100, verbose=0)

    # 预测
    yhat_train = model.predict(X_train, verbose=0)
    yhat_test = model.predict(X_test, verbose=0)

    # 反归一化
    yhat_train_rescaled = scaler.inverse_transform(yhat_train)
    y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))
    yhat_test_rescaled = scaler.inverse_transform(yhat_test)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 计算 MSE
    mse_train = mean_squared_error(y_train_rescaled, yhat_train_rescaled)
    mse_test = mean_squared_error(y_test_rescaled, yhat_test_rescaled)
    print(f'Training Mean Squared Error: {mse_train}')
    print(f'Test Mean Squared Error: {mse_test}')

    # 绘制预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_train_rescaled)), y_train_rescaled, label='Actual Train')
    plt.plot(np.arange(len(y_train_rescaled)), yhat_train_rescaled, label='Predicted Train')
    plt.plot(np.arange(len(y_train_rescaled), len(y_train_rescaled) + len(y_test_rescaled)), y_test_rescaled, label='Actual Test')
    plt.plot(np.arange(len(y_train_rescaled), len(y_train_rescaled) + len(y_test_rescaled)), yhat_test_rescaled, label='Predicted Test')
    plt.title('LSTM Time Series Prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# 使用函数
# tapes_df = pd.read_csv('path_to_your_data.csv')  # 加载数据
# process_and_predict(tapes_df)
