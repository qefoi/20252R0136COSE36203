#pip install pykrx 
#pip install statsmodels
#pip install scikit-learn
#pip install tabulate
#pip install tensorflow

from pykrx import stock
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tabulate
import matplotlib.pyplot as plt

start_date = "20130101"
end_date = "20241231"
ticker_code = "005930" 
window_size = 5
TIMESTEPS = 10

# 재현성 확보
tf.random.set_seed(42)
np.random.seed(42)

OHLCV = stock.get_market_ohlcv_by_date(start_date, end_date, ticker_code)
data = OHLCV[['시가', '고가', '저가', '종가', '거래량']]
data.columns = ['open', 'high', 'low', 'close', 'volume']

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

data['body'] = abs((data['close'] - data['open']) / data['open']) * 100
data['upper_shadow'] = (data['high'] - data[['open', 'close']].max(axis=1)) / data['open'] * 100
data['lower_shadow'] = (data[['open', 'close']].min(axis=1) - data['low']) / data['open'] * 100
data['body_ratio'] = safe_divide(abs(data['close'] - data['open']), (data['high'] - data['low']))
data['body_ratio'] = data['body_ratio'].replace([np.inf, -np.inf], 0).fillna(0)

shadow_diff = data['upper_shadow'] - data['lower_shadow']
high_low_diff_norm = safe_divide(data['high'] - data['low'], data['open']) * 100
data['shadow_ratio'] = safe_divide(shadow_diff, high_low_diff_norm)
data['shadow_ratio'] = data['shadow_ratio'].replace([np.inf, -np.inf], 0).fillna(0)

data['direction'] = np.sign(data['close'] - data['open'])
data['volume_strength'] = data['volume'] / data['volume'].rolling(5).mean()
data['momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) * 100

data[['next_open', 'next_high', 'next_low', 'next_close', 'next_volume']] = data[['open', 'high', 'low', 'close', 'volume']].shift(-1)

data['target_open_chg'] = (data['next_open'] - data['open']) / data['open'] * 100
data['target_high_chg'] = (data['next_high'] - data['high']) / data['high'] * 100
data['target_low_chg'] = (data['next_low'] - data['low']) / data['low'] * 100
data['target_close_chg'] = (data['next_close'] - data['close']) / data['close'] * 100
data['target_volume_chg'] = np.log1p(data['next_volume'] / data['volume']) - np.log1p(1)

target_cols = ['target_open_chg', 'target_high_chg', 'target_low_chg', 'target_close_chg', 'target_volume_chg']

feature_column_base = ['body', 'upper_shadow', 'lower_shadow', 'body_ratio', 
                       'shadow_ratio', 'direction', 'volume_strength', 'momentum']

final_cols = feature_column_base + target_cols
combined_data = data[final_cols].copy()
combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
combined_data.dropna(inplace=True)

x_data = combined_data[feature_column_base]
y_data = combined_data[target_cols]

# 입력 데이터 정규화
scaler_x = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler_x.fit_transform(x_data)

# 출력 데이터 정규화
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(y_data)

# 2D 데이터를 3D 시퀀스 데이터로 변환하는 함수 (LSTM 입력 형태)
def create_sequences(features, targets, time_steps):
    X, Y = [], []
    for i in range(len(features) - time_steps):
        # X: i 시점부터 time_steps 길이만큼의 과거 데이터
        X.append(features[i:(i + time_steps)])
        # Y: i + time_steps 시점의 타겟 데이터 (바로 다음 날)
        Y.append(targets[i + time_steps])
    return np.array(X), np.array(Y)

# 시퀀스 데이터 생성 (3D 배열)
X_seq, Y_seq = create_sequences(x_scaled, y_scaled, TIMESTEPS)

train_size = int(len(X_seq) * 0.7)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
Y_train, Y_test = Y_seq[:train_size], Y_seq[train_size:] 

# LSTM 신경망 구조 (Sequence Learning) 
model = Sequential([
    # LSTM 층: 시퀀스 정보를 학습하는 핵심 레이어
    LSTM(units=50, activation='tanh', return_sequences=True, 
         input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2), # 과적합 방지
    LSTM(units=50, activation='tanh'),
    Dropout(0.2),
    # Dense 층: 최종적으로 5개의 타겟 변수를 출력 
    Dense(len(target_cols)) 
])

# mse와 adam 사용
model.compile(optimizer='adam', loss='mse')

# 학습 실행
history = model.fit(
    X_train, Y_train,
    epochs=50, 
    batch_size=32,
    validation_split=0.1, # 학습 데이터 중 10%를 검증에 사용
    verbose=0
)

# 예측 실행
Y_predict_scaled = model.predict(X_test, verbose=0)

# 예측 결과를 원래 변화율로 역변환
Y_predict = scaler_y.inverse_transform(Y_predict_scaled)
Y_test_actual = scaler_y.inverse_transform(Y_test)

# 데이터프레임으로 변환하여 정리
sequence_indices = x_data.index[TIMESTEPS:]
test_indices = sequence_indices[train_size:]

Y_predict_df = pd.DataFrame(Y_predict, columns=target_cols, index=test_indices)
Y_test_df = pd.DataFrame(Y_test_actual, columns=target_cols, index=test_indices)

rmse_results = {}
r2_results = {}
report_data = []

for col in target_cols:
    actual_chg = Y_test_df[col]
    predicted_chg = Y_predict_df[col]
    
    # RMSE 
    rmse = np.sqrt(mean_squared_error(actual_chg, predicted_chg))
    rmse_results[col] = rmse
    
    # R-squared
    r2 = r2_score(actual_chg, predicted_chg)
    r2_results[col] = r2

    display_name = col.replace('target_', '').replace('_chg', '').upper()
    report_data.append([f"Next {display_name}", f"{rmse:.4f}", f"{r2:.6f}"])

headers = ["예측 항목", "RMSE (변화율)", "R-squared (변화율)"]
print(tabulate.tabulate(report_data, headers=headers, tablefmt="markdown"))
# 시각화

fig, axes = plt.subplots(5, 1, figsize=(16, 15), sharex=True)
plot_titles = ['Open Change (%)', 'High Change (%)', 'Low Change (%)', 'Close Change (%)', 'Volume Log Change']

for i, col in enumerate(target_cols):
    ax = axes[i]
    
    # 실제 변화율
    ax.plot(Y_test_df.index, Y_test_df[col], label='Actual Change Rate', color='blue', linewidth=1)
    
    # 예측 변화율
    ax.plot(Y_predict_df.index, Y_predict_df[col], label='Predicted Change Rate', color='red', linestyle='--', linewidth=1)
    
    # 제목 및 지표 표시
    rmse_val = rmse_results[col]
    r2_val = r2_results[col]
    ax.set_title(f"{plot_titles[i]} (RMSE: {rmse_val:.4f}, R^2: {r2_val:.6f})", fontsize=12)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8) # 0% 기준선
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.5)

plt.xlabel('Date')
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
plt.show()
