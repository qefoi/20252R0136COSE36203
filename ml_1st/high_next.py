#pip install pykrx 
#pip install statsmodels
#pip install scikit-learn

from pykrx import stock
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

start_date = "20200101"
end_date = "20241231"
ticker_code = "005930" # 삼성전자
window_size = 2

OHLCV = stock.get_market_ohlcv_by_date(start_date, end_date, ticker_code)
data = OHLCV[['시가', '고가', '저가', '종가', '거래량']]
data.columns = ['open', 'high', 'low', 'close', 'volume']

# 오늘의 데이터를 이용해 다음날의 고가를 예측해야하므로 하루씩 뒤로 밀기
data['next_high'] = data['high'].shift(-1)

# 5일전까지의 data를 새로운 column으로 추가
target_column = ['open', 'high', 'low', 'close']
for col in target_column:
    for i in range(1, window_size + 1):
        data[f'{col}_{i}_days_ago'] = data[col].shift(i)

# NaN 포함되는 데이터 삭제(앞에서 window_size만큼의 data, 마지막 data)
data.dropna(inplace=True)

x = data.filter(regex='_days_ago$')
y = data['next_high']
x = sm.add_constant(x)
print(f"x 변수 개수 (상수항 포함): {x.shape[1]}")

# 70% -> training, 30% -> test
train_size = int(len(x) * 0.7)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:] 

model = sm.OLS(y_train, x_train)
result = model.fit()

print("\n--- 회귀 모델 학습 결과 ---")
print(result.summary())

# 테스트셋으로 테스트
y_predict = result.predict(x_test)

mse = np.mean((y_test - y_predict) ** 2) 
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

result_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_predict}, index=x_test.index)
plt.figure(figsize=(14, 7))
plt.plot(result_df.index, result_df['Actual'], label='Actual Next High', color='blue', linewidth=2)
plt.plot(result_df.index, result_df['Predicted'], label='Predicted Next High', color='red', linestyle='--', linewidth=2)

plt.title('Test Set: Actual vs. Predicted Next High Price (Regression)')
plt.xlabel('Date')
plt.ylabel('Price (KRW)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.text(
    0.01,  # X 좌표 (0.0 ~ 1.0)
    0.95,  # Y 좌표 (0.0 ~ 1.0)
    f'RMSE: {rmse:,.2f} KRW', # 출력할 텍스트
    transform=plt.gca().transAxes, # Axes 좌표계를 사용하도록 설정
    fontsize=14, 
    color='darkgreen', 
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
)
plt.tight_layout()
plt.show()




