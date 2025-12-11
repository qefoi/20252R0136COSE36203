# pip install tensorflow pandas numpy scikit-learn matplotlib pykrx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrx import stock
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# 랜덤 시드 고정 (재현성 확보)
np.random.seed(42)
tf.random.set_seed(42)

start_date = "20130101" # 데이터 수를 늘리기 위해 2013년부터 사용
end_date = "20241231"
ticker_code = "005930" # 삼성전자
window_size = 5

OHLCV = stock.get_market_ohlcv_by_date(start_date, end_date, ticker_code)
data = OHLCV[['시가', '고가', '저가', '종가', '거래량']]
data.columns = ['open', 'high', 'low', 'close', 'volume']
data = data[data['volume'] > 0]

# CLV 계산
data['clv'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
data['clv'] = data['clv'].fillna(0)

feature_cols = ['open', 'high', 'low', 'close', 'volume', 'clv']
label_col = 'target'

# Target
data['price_change'] = data['close'] - data['close'].shift(1)
data['target'] = np.where(data['price_change'] > 0, 1, 0)

feature_values = data[feature_cols].values
label_values = data[label_col].values

# 시계열 데이터셋 구성 함수
# X: [t-n ... t-1] (과거 5일치)
# y: [t] (오늘의 등락)
def make_lstm_dataset(features, labels, window_size):
    X_list, y_list = [], []
    for i in range(window_size, len(features)):
        # i 시점의 타겟을 예측하기 위해 i-window ~ i-1 까지의 데이터를 사용
        X_list.append(features[i - window_size : i])
        y_list.append(labels[i])
    return np.array(X_list), np.array(y_list)

# 시계열 순서를 지키며 분할
train_size = int(len(feature_values) * 0.7)

# 스케일링
scaler = StandardScaler()
train_features_raw = feature_values[:train_size]
scaler.fit(train_features_raw)

scaled_features = scaler.transform(feature_values)

# LSTM 입력 데이터 생성
X, y = make_lstm_dataset(scaled_features, label_values, window_size)

# 만들어진 X, y를 다시 Train/Test로 분할
# make_lstm_dataset 과정에서 앞쪽 window_size만큼 데이터가 빠지므로 인덱스 재조정 필요
split_idx = train_size - window_size 

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"입력 데이터 형태(X_train): {X_train.shape}")
print(f"타겟 데이터 형태(y_train): {y_train.shape}")

# LSTM 모델 구축
model = Sequential()

# LSTM Layer
# return_sequences=False: 마지막 타임스텝의 결과만 다음 레이어로 전달 (Many-to-One)
model.add(LSTM(units=64, input_shape=(window_size, len(feature_cols)), activation='tanh'))
model.add(Dropout(0.2)) # 과적합 방지

# Dense Layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Output Layer (0 or 1 분류이므로 sigmoid)
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# 모델 학습
# EarlyStopping: 검증 손실이 10번 이상 개선되지 않으면 학습 조기 종료
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_split=0.2, 
                    callbacks=[early_stop],
                    verbose=1)

# 평가 및 결과 시각화
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*40)
print(f"  [ LSTM 모델 예측 결과 ]")
print("="*40)
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(f"TN(하락 맞춤): {cm[0][0]} | FP(틀린 상승): {cm[0][1]}")
print(f"FN(틀린 하락): {cm[1][0]} | TP(상승 맞춤): {cm[1][1]}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fall', 'Rise']))

# 학습 과정 시각화 (Loss 추이)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (Cross Entropy)')
plt.legend()

# 실제 vs 예측 비교 (최근 100일)
plt.subplot(1, 2, 2)
subset_len = 100
plt.plot(y_test[-subset_len:], 'b.', label='Actual', alpha=0.6)
plt.plot(y_pred_proba[-subset_len:], 'r-', label='Predicted Prob', alpha=0.6)
plt.axhline(0.5, color='gray', linestyle='--')
plt.title('Prediction Probability vs Actual (Last 100 days)')
plt.legend()

plt.tight_layout()
plt.show()

# for ROC-AUC Score
y_pred_proba = model.predict(X_test).flatten()

roc_score = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*40)
print(f"ROC-AUC Score: {roc_score:.4f}")
print("="*40)