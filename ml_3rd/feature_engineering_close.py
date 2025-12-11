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

data['body'] = abs((data['close'] - data['open']) / data['open']) * 100
data['upper_shadow'] = (data['high'] - data[['open', 'close']].max(axis=1)) / data['open'] * 100
data['lower_shadow'] = (data[['open', 'close']].min(axis=1) - data['low']) / data['open'] * 100

# 0으로 나누는 위험 방지용 epsilon
eps = 1e-9
range_hl = (data['high'] - data['low']).replace(0, np.nan)

data['body_ratio'] = abs(data['close'] - data['open']) / (range_hl + eps)
data['shadow_ratio'] = (data['upper_shadow'] - data['lower_shadow']) / (((range_hl / data['open']) * 100) + eps)

data['direction'] = np.sign(data['close'] - data['open'])
data['volume_strength'] = data['volume'] / data['volume'].rolling(5).mean()
data['momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) * 100

data['target_change'] = ((data['close'].shift(-1) - data['close']) / data['close']) * 100 

data['current_close_level'] = data['close']
data['actual_next_close_level'] = data['close'].shift(-1)

# window size전까지의 data를 새로운 column으로 추가
target_column = ['body', 'upper_shadow', 'lower_shadow', 'body_ratio', 
                   'shadow_ratio', 'direction', 'volume_strength', 'momentum']

x_features = pd.DataFrame(index=data.index)
for col in target_column:
    for i in range(1, window_size + 1):
        x_features[f'{col}_{i}_days_ago'] = data[col].shift(i)

# 1. X와 Y를 하나의 DataFrame으로 합친 후, NaN을 일괄 제거
# data = pd.concat([x_features, data['target_change']], axis=1)
data = pd.concat([x_features, data[['target_change', 'current_close_level', 'actual_next_close_level']]], axis=1)
data.dropna(inplace=True)

# 2. 깨끗한 X와 Y 분리
# x = data.drop(columns=['target_change'])
x = data.drop(columns=['target_change', 'current_close_level', 'actual_next_close_level'])
y = data['target_change'] # Y는 변화량 (target_change)
x = sm.add_constant(x)
print(f"예측 모델의 x 변수 개수 (상수항 포함): {x.shape[1]}")

# 70% -> training, 30% -> test
train_size = int(len(x) * 0.7)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test_change = y[:train_size], y[train_size:] 

# 테스트셋의 레벨 데이터 추출
test_data_levels = data.loc[x_test.index]
current_close_level_test = test_data_levels['current_close_level']
actual_next_close_level_test = test_data_levels['actual_next_close_level']


model_change = sm.OLS(y_train, x_train)
result_change = model_change.fit()

print("\n" + "=" * 80)
print("1단계: 가격 변화량 예측 모델 학습 결과")
print("=" * 80)
print(f"Condition Number: {result_change.condition_number:.2e}")
print(f"R-squared: {result_change.rsquared:.4f}")
print(result_change.summary().as_text())

y_predict_change = result_change.predict(x_test)
change_rmse = np.sqrt(mean_squared_error(y_test_change, y_predict_change))
change_r2 = r2_score(y_test_change, y_predict_change)

# 레벨 복원 및 평가
# Predicted Level = Current Level * (1 + Predicted % Change / 100)
predicted_close_level = current_close_level_test * (1 + y_predict_change / 100)
level_rmse = np.sqrt(mean_squared_error(actual_next_close_level_test, predicted_close_level))
level_r2 = r2_score(actual_next_close_level_test, predicted_close_level)

print("\n" + "=" * 80)
print("2단계: 테스트셋 적용 결과")
print("=" * 80)
print(f"Condition Number: {result_change.condition_number:.2e}")
print("\n--- 변화율 (Change Rate, %) 예측 성능 ---")
print(f"Close Change Rate RMSE: {change_rmse:.4f} %")
print(f"Close Change Rate R^2: {change_r2:.4f}")
print("\n--- 절대 가격 (Level) 복원 성능 ---")
print(f"Close Level RMSE: {level_rmse:.4f} KRW")
print(f"Close Level R^2 (추세 반영): {level_r2:.4f}")

# 시각화 1: 가격 변화율 예측 (통계적 유효성 확인)
result_change_df = pd.DataFrame({'Actual': y_test_change.values, 'Predicted': y_predict_change}, index=x_test.index)
plt.figure(figsize=(16, 7))
# 실제 변화율 (%) vs 예측 변화율 (%)
plt.plot(result_change_df.index, result_change_df['Actual'], label='Actual Next Close Change Rate', color='blue', linewidth=2)
plt.plot(result_change_df.index, result_change_df['Predicted'], label='Predicted Next Close Change Rate', color='green', linestyle='--', linewidth=2)

plt.title('1. Price Change Rate Prediction (Target: % Change)')
plt.xlabel('Date')
plt.ylabel('Price Change Rate (%)') # Y축 단위 %로 수정
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.text(
    0.01, 0.90, f'Change Rate R²: {change_r2:.4f}', 
    transform=plt.gca().transAxes, fontsize=12, color='darkred'
)
plt.text(
    0.01, 0.95, f'Change Rate RMSE: {change_rmse:,.2f} %', # 단위 %로 수정
    transform=plt.gca().transAxes, fontsize=14, color='darkgreen'
)
plt.tight_layout()
plt.show()


# 시각화 2: 절대 가격 레벨 예측 (가짜 성공 시각화)
result_level_df = pd.DataFrame({'Actual': actual_next_close_level_test.values, 'Predicted': predicted_close_level.values}, index=x_test.index)

plt.figure(figsize=(16, 7))
plt.plot(result_level_df.index, result_level_df['Actual'], label='Actual Next Close Level', color='blue', linewidth=2)
plt.plot(result_level_df.index, result_level_df['Predicted'], label='Predicted Next Close Level', color='red', linestyle='--', linewidth=2)

plt.title('2. close Level')
plt.xlabel('Date')
plt.ylabel('Close Level (KRW)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.text(
    0.01, 0.90, f'Level R²: {level_r2:.4f}', 
    transform=plt.gca().transAxes, fontsize=12, color='darkred'
)
plt.text(
    0.01, 0.95, f'Level RMSE: {level_rmse:,.0f} KRW', 
    transform=plt.gca().transAxes, fontsize=14, color='darkgreen'
)
plt.tight_layout()
plt.show()