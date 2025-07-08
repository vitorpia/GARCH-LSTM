import os
import time
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#%% Output folder (change this to your preferred path)
output_path = r"C:\Your\Preferred\Path\to\Save\LSTM_Graphs"
os.makedirs(output_path, exist_ok=True)

#%% Download data
start = '2010-01-01'
end = '2025-01-01'
df = yf.download('USDBRL=X', start=start, end=end)[['Close']].rename(columns={'Close': 'BRL'}).dropna()

#%% Returns and realized volatility
ret = np.log(df / df.shift(1)).dropna()
rv = ret.rolling(21).std().dropna() * np.sqrt(252)
rv.columns = ['RV']

#%% Fit GARCH-type models
def fit_garch(y, model_type='GARCH'):
    if model_type == 'GARCH':
        model = arch_model(y, vol='GARCH', p=1, q=1, dist='normal')
    elif model_type == 'EGARCH':
        model = arch_model(y, vol='EGARCH', p=1, q=1, dist='normal')
    elif model_type == 'GJR':
        model = arch_model(y, vol='GARCH', p=1, o=1, q=1, dist='normal')
    elif model_type == 'APARCH':
        model = arch_model(y, vol='APARCH', p=1, o=1, q=1, dist='normal')
    else:
        raise ValueError("Unknown GARCH model type.")
    start_time = time.time()
    res = model.fit(disp="off")
    elapsed = time.time() - start_time
    return res.conditional_volatility * np.sqrt(252), elapsed

#%% Train LSTM
def train_lstm(X, y):
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, verbose=0, callbacks=[early])
    return model

#%% Save diagnostic plots
def save_diagnostics(y_true, y_pred, model_name):
    residuals = y_true.flatten() - y_pred.flatten()
    abs_error = np.abs(residuals)

    # 1. Residuals over time
    plt.figure(figsize=(12, 4))
    plt.plot(residuals)
    plt.title("Prediction Residuals")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_path}/{model_name}_residuals.png")
    plt.close()

    # 2. Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=30, color="orange", edgecolor='black')
    plt.title("Residuals Distribution")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_path}/{model_name}_residuals_hist.png")
    plt.close()

    # 3. ACF
    plt.figure(figsize=(6, 4))
    plot_acf(residuals, lags=40)
    plt.title("Residuals ACF")
    plt.tight_layout()
    plt.savefig(f"{output_path}/{model_name}_acf.png")
    plt.close()

    # 4. PACF
    plt.figure(figsize=(6, 4))
    plot_pacf(residuals, lags=40, method='ywm')
    plt.title("Residuals PACF")
    plt.tight_layout()
    plt.savefig(f"{output_path}/{model_name}_pacf.png")
    plt.close()

    # 5. Scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Realized Volatility")
    plt.ylabel("LSTM Prediction")
    plt.title("RV vs. LSTM Prediction")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_path}/{model_name}_scatter.png")
    plt.close()

    # 6. Absolute Error
    plt.figure(figsize=(12, 4))
    plt.plot(abs_error)
    plt.title("Absolute Prediction Error")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_path}/{model_name}_abs_error.png")
    plt.close()

#%% Main loop
results = []

for garch_type in ['GARCH', 'EGARCH', 'GJR', 'APARCH']:
    sigma, garch_time = fit_garch(ret['BRL'], model_type=garch_type)
    df_lstm = pd.concat([sigma.rename('vol'), rv], axis=1).dropna()

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(df_lstm[['vol']])
    y_scaled = scaler_y.fit_transform(df_lstm[['RV']])

    window = 30
    X_seq, y_seq = [], []
    for i in range(window, len(df_lstm)):
        X_seq.append(X_scaled[i-window:i])
        y_seq.append(y_scaled[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    model_lstm = train_lstm(X_seq, y_seq)
    y_pred = model_lstm.predict(X_seq)
    y_true = scaler_y.inverse_transform(y_seq)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred_inv))
    r2 = r2_score(y_true, y_pred_inv)

    results.append({
        'GARCH_Model': garch_type,
        'RMSE_LSTM': rmse,
        'R2_LSTM': r2,
        'GARCH_Time': garch_time
    })

    # Main forecast plot
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label='Realized Volatility')
    plt.plot(y_pred_inv, label=f'LSTM Prediction - {garch_type}')
    plt.title(f'{garch_type} + LSTM | RMSE: {rmse:.5f} | R²: {r2:.3f}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_path}/{garch_type}_forecast.png")
    plt.close()

    # Save diagnostics
    save_diagnostics(y_true, y_pred_inv, garch_type)

#%% Print results
df_results = pd.DataFrame(results)
print(df_results)