#pip install pykrx 
#pip install statsmodels
#pip install scikit-learn
#pip install tabulate

from pykrx import stock
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import tabulate

start_date = "20130101"
end_date = "20241231"
ticker_code = "005930" # 삼성전자
window_size = 2

OHLCV = stock.get_market_ohlcv_by_date(start_date, end_date, ticker_code)
data = OHLCV[['시가', '고가', '저가', '종가', '거래량']]
data.columns = ['open', 'high', 'low', 'close', 'volume']

# 1. body (당일 봉 몸통 길이, %)
data['body'] = ((data['close'] - data['open']).abs() / data['open']) * 100

# 2. upper_shadow (윗꼬리 길이, %)
data['upper_shadow'] = (data['high'] - data[['open', 'close']].max(axis=1)) / data['open'] * 100

# 3. lower_shadow (아랫꼬리 길이, %)
data['lower_shadow'] = (data[['open', 'close']].min(axis=1) - data['low']) / data['open'] * 100

# 4. body_ratio (몸통이 전체 봉(high-low)에서 차지하는 비율) [cite: 21]
# abs(close - open) / (high - low)
data['body_ratio'] = (data['close'] - data['open']).abs() / (data['high'] - data['low'])
# 분모가 0인 경우 처리 (high == low)
data['body_ratio'] = data['body_ratio'].replace([np.inf, -np.inf], 0).fillna(0)

# 5. shadow_ratio (꼬리의 불균형 정도)
# 윗꼬리+아랫꼬리 합 / (고가-저가) -> 보고서 정의와 유사하게 비율로 계산
data['shadow_ratio'] = (data['upper_shadow'] - data['lower_shadow']) / (data['high'] - data['low'] / data['open'] * 100)
data['shadow_ratio'] = data['shadow_ratio'].replace([np.inf, -np.inf], 0).fillna(0)

# 6. direction (상승 +1 / 하락 -1 방향)
data['direction'] = np.sign(data['close'] - data['open'])

# 7. volume_strength (최근 5일 평균 대비 거래량 강도)
data['volume_strength'] = data['volume'] / data['volume'].rolling(window=5).mean()

# 8. momentum (전일 대비 종가 상승률, %)
data['momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) * 100

data['target_change'] = data['close'].shift(-1) - data['close'] # 학습 Y

# window size전까지의 data를 새로운 column으로 추가
target_column = ['body', 'upper_shadow', 'lower_shadow', 'body_ratio', 
                   'shadow_ratio', 'direction', 'volume_strength', 'momentum']

x_features = pd.DataFrame(index=data.index)
for col in target_column:
    for i in range(1, window_size + 1):
        x_features[f'{col}_{i}_days_ago'] = data[col].shift(i)

# 1. X와 Y를 하나의 DataFrame으로 합친 후, NaN을 일괄 제거
data = pd.concat([x_features, data['target_change']], axis=1)
data.dropna(inplace=True)

# 2. 깨끗한 X와 Y 분리
x = data.drop(columns=['target_change'])
y = data['target_change'] # Y는 변화량 (target_change)
x = sm.add_constant(x)
print(f"예측 모델의 x 변수 개수 (상수항 포함): {x.shape[1]}")

# 70% -> training, 30% -> test
train_size = int(len(x) * 0.7)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test_change = y[:train_size], y[train_size:] 

model_change = sm.OLS(y_train, x_train)
result_change = model_change.fit()

print("\n" + "=" * 80)
print("1단계: 가격 변화량 예측 모델 학습 결과")
print("=" * 80)
print(f"Condition Number: {result_change.condition_number:.2e}")
print(f"R-squared: {result_change.rsquared:.4f}")
print(result_change.summary().as_text())

y_predict_change = result_change.predict(x_test)
final_rmse = np.sqrt(mean_squared_error(y_test_change, y_predict_change))
final_r2 = r2_score(y_test_change, y_predict_change)

print("\n" + "=" * 80)
print("2단계: 테스트셋 적용 결과")
print("=" * 80)
print(f"Final Absolute Price RMSE: {final_rmse:.4f} KRW")
print(f"Final Absolute Price R^2: {final_r2:.4f}")

# --- 시각화 (가격 변화량 예측) ---
result_df = pd.DataFrame({'Actual': y_test_change.values, 'Predicted': y_predict_change}, index=x_test.index)
plt.figure(figsize=(14, 7))
plt.plot(result_df.index, result_df['Actual'], label='Actual Next Close (Level)', color='blue', linewidth=2)
plt.plot(result_df.index, result_df['Predicted'], label='Predicted Next Close (Level)', color='green', linestyle='--', linewidth=2)

plt.title('Final Prediction: Absolute Price (Level) using Change Model')
plt.xlabel('Date')
plt.ylabel('Price (KRW)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.text(
    0.01, 
    0.95, 
    f'Final RMSE: {final_rmse:,.2f} KRW', 
    transform=plt.gca().transAxes, 
    fontsize=14, 
    color='darkgreen', 
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
)
plt.tight_layout()
plt.show()