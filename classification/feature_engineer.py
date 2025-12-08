from pykrx import stock
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
start_date = "20220101"
end_date = "20241231"
ticker_code = "005930"  # 삼성전자
window_size = 5

OHLCV = stock.get_market_ohlcv_by_date(start_date, end_date, ticker_code)
data = OHLCV[['시가', '고가', '저가', '종가', '거래량']]
data.columns = ['open', 'high', 'low', 'close', 'volume']

# 2. Feature Engineering
data['body'] = abs((data['close'] - data['open']) / data['open']) * 100
data['upper_shadow'] = (data['high'] - data[['open', 'close']].max(axis=1)) / data['open'] * 100
data['lower_shadow'] = (data[['open', 'close']].min(axis=1) - data['low']) / data['open'] * 100
data['body_ratio'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
data['shadow_ratio'] = (data['upper_shadow'] - data['lower_shadow']) / ((data['high'] - data['low']) / data['open'] * 100)
data['direction'] = np.sign(data['close'] - data['open'])
data['volume_strength'] = data['volume'] / data['volume'].rolling(5).mean()
data['momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) * 100

# 3. 라벨링 (3-class)
data['next_close'] = data['close'].shift(-1)
data['return'] = (data['next_close'] - data['close']) / data['close'] * 100

threshold = 0.3  # ±0.3% 이내는 변동 미미로 간주
conditions = [
    data['return'] < -threshold,                 # 하락
    (data['return'] >= -threshold) & (data['return'] <= threshold),  # 변동 미미
    data['return'] > threshold                   # 상승
]
choices = [0, 1, 2]
data['label'] = np.select(conditions, choices)

# 4. 시계열 feature 추가
features = ['body', 'upper_shadow', 'lower_shadow', 'body_ratio',
            'shadow_ratio', 'direction', 'volume_strength', 'momentum']

for col in features:
    for i in range(1, window_size + 1):
        data[f'{col}_{i}_days_ago'] = data[col].shift(i)

data.dropna(inplace=True)

# 5. Train/Test Split
x = data.filter(regex='_days_ago$')
y = data['label']

train_size = int(len(x) * 0.7)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 6. Multinomial Logistic Regression
model = LogisticRegression(max_iter=1000, multi_class='multinomial')
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)
y_prob = model.predict_proba(x_test_scaled)

# 7. 평가
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

# 8. 시각화
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_prob[:, 2], color='blue', alpha=0.6, label='Up Probability')
plt.plot(y_test.index, y_prob[:, 0], color='red', alpha=0.6, label='Down Probability')
plt.plot(y_test.index, y_prob[:, 1], color='gray', alpha=0.4, label='Neutral Probability')

# 실제 구간 색상 표시
colors = y_test.map({0: 'red', 1: 'gray', 2: 'blue'})
plt.scatter(y_test.index, [1]*len(y_test), c=colors, s=10, alpha=0.6, label='Actual Class (red=down, gray=neutral, blue=up)')

plt.title('Predicted Class Probabilities (Down / Neutral / Up)')
plt.xlabel('Date')
plt.ylabel('Predicted Probability')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
