# pip install pykrx scikit-learn pandas numpy matplotlib seaborn

from pykrx import stock
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start_date = "20190101"
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

# Target
data['price_change'] = data['close'] - data['close'].shift(1)
data['target'] = np.where(data['price_change'] > 0, 1, 0)

feature_cols = ['open', 'high', 'low', 'close', 'volume', 'clv']

for col in feature_cols:
    for i in range(1, window_size + 1):
        data[f'{col}_{i}d'] = data[col].shift(i)

# NaN 포함되는 데이터 삭제(앞에서 window_size만큼의 data, 마지막 data)
data.dropna(inplace=True)

# '_d'로 끝나는 과거 데이터만 X로 설정
X = data.filter(regex='_\d+d$')
y = data['target']

# 데이터 정렬 확인용 출력
print("[데이터 정렬 검증]")
print(f"Row Date(Today): {X.index[-1]}")
print(f"Target (Today's Result): {y.iloc[-1]} (1=Rise, 0=Fall)")
print(f"Feature (1 day ago close): {X['close_1d'].iloc[-1]}")
print(f"Actual Previous Close: {OHLCV['종가'].iloc[-2]}")
print("-" * 40)

train_size = int(len(X) * 0.7)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# 스케일링 & 모델 학습
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train_scaled, y_train)

# 평가 및 결과 확인
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n[ 최종 모델 평가 ]")
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(f"TN(하락 맞춤): {cm[0][0]} | FP(틀린 상승): {cm[0][1]}")
print(f"FN(틀린 하락): {cm[1][0]} | TP(상승 맞춤): {cm[1][1]}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Fall', 'Rise']))

plt.figure(figsize=(14, 6))
plt.plot(y_test.index, data.loc[y_test.index, 'close'], color='gray', alpha=0.5, label='Price')

# 맞춘 곳(Green)과 틀린 곳(Red) 산점도
correct_mask = (y_test == y_pred)
plt.scatter(y_test[correct_mask].index, data.loc[y_test[correct_mask].index, 'close'], 
            c='green', s=20, label='Correct Prediction')
plt.scatter(y_test[~correct_mask].index, data.loc[y_test[~correct_mask].index, 'close'], 
            c='red', marker='x', s=20, label='Wrong Prediction')

plt.title('Prediction Result based on Previous n-days Data')
plt.legend()
plt.show()