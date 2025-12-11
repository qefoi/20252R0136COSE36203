# pip install pykrx scikit-learn pandas numpy matplotlib

from pykrx import stock
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# 매수/매도 압력 대리 지표 (CLV)
data['clv'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
data['clv'] = data['clv'].fillna(0)

# 오늘의 종가가 어제보다 상승했는지 여부 (Target)
data['price_change'] = data['close'] - data['close'].shift(1)
data['target'] = np.where(data['price_change'] > 0, 1, 0)

# 이전 n일 동안의 데이터로 컬럼 생성
feature_cols = ['open', 'high', 'low', 'close', 'volume', 'clv']
for col in feature_cols:
    for i in range(1, window_size + 1):
        data[f'{col}_{i}d'] = data[col].shift(i)

data.dropna(inplace=True)

X = data.filter(regex='_\d+d$')
y = data['target']

# 시계열 순서 유지하며 분할
train_size = int(len(X) * 0.7)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest 모델 설정 및 학습
model = RandomForestClassifier(
    n_estimators=200,    # 나무의 개수
    max_depth=10,        # 과적합 방지를 위한 깊이 제한
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# 테스트셋 예측
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {roc:.4f}")
print("-" * 30)
print(f"Confusion Matrix:\n{cm}")

# 결과 시각화
plt.figure(figsize=(14, 6))
plt.plot(y_test.values[-100:], 'b.', label='Actual', alpha=0.3)
plt.plot(y_pred_proba[-100:], 'r-', label='Predicted Prob', alpha=0.8)
plt.axhline(0.5, color='gray', linestyle='--')
plt.title('Random Forest Prediction vs Actual (Last 100 days)')
plt.legend()
plt.tight_layout()
plt.show()