#pip install pykrx 
#pip install statsmodels
#pip install scikit-learn
#pip install tabulate
#pip install lightgbm

from pykrx import stock
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import tabulate
import matplotlib.pyplot as plt

# 데이터 불러오기
start_date = "20130101"
end_date = "20241231"
ticker_code = "005930" # 삼성전자
window_size = 5

OHLCV = stock.get_market_ohlcv_by_date(start_date, end_date, ticker_code)
data = OHLCV[['시가', '고가', '저가', '종가', '거래량']]
data.columns = ['open', 'high', 'low', 'close', 'volume']

# Feature Engineering
data['body'] = abs((data['close'] - data['open']) / data['open']) * 100
data['upper_shadow'] = (data['high'] - data[['open', 'close']].max(axis=1)) / data['open'] * 100
data['lower_shadow'] = (data[['open', 'close']].min(axis=1) - data['low']) / data['open'] * 100
data['body_ratio'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
data['shadow_ratio'] = (data['upper_shadow'] - data['lower_shadow']) / ((data['high'] - data['low']) / data['open'] * 100)
data['direction'] = np.sign(data['close'] - data['open'])
data['volume_strength'] = data['volume'] / data['volume'].rolling(5).mean()
data['momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) * 100

# 다음 날 OHLCV 레벨 정의
data[['next_open', 'next_high', 'next_low', 'next_close', 'next_volume']] = data[['open', 'high', 'low', 'close', 'volume']].shift(-1)

# 변화율 변수 정의
data['target_open_chg'] = (data['next_open'] - data['open']) / data['open'] * 100
data['target_high_chg'] = (data['next_high'] - data['high']) / data['high'] * 100
data['target_low_chg'] = (data['next_low'] - data['low']) / data['low'] * 100
data['target_close_chg'] = (data['next_close'] - data['close']) / data['close'] * 100
# 거래량: Log를 사용하여 큰 변동성 처리
data['target_volume_chg'] = np.log1p(data['next_volume'] / data['volume']) - np.log1p(1)

# 타겟 목록
target_cols = ['target_open_chg', 'target_high_chg', 'target_low_chg', 'target_close_chg', 'target_volume_chg']


feature_column_base = ['body', 'upper_shadow', 'lower_shadow', 'body_ratio', 
                       'shadow_ratio', 'direction', 'volume_strength', 'momentum']

x_features = pd.DataFrame(index=data.index)
for col in feature_column_base:
    for i in range(1, window_size + 1):
        x_features[f'{col}_{i}_days_ago'] = data[col].shift(i)


# NaN 포함 데이터 삭제
data.dropna(inplace=True)
x = x_features.loc[data.index].dropna()
y = data[target_cols].loc[x.index] # Y는 5개 변화율 컬럼을 가진 DataFrame

train_size = int(len(x) * 0.7)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:] 

# 테스트셋의 현재 레벨 정보를 추출 (RMSE 계산을 위해 필요)
current_level_test = data[['open', 'high', 'low', 'close', 'volume']].loc[x_test.index]
next_level_test = data[['next_open', 'next_high', 'next_low', 'next_close', 'next_volume']].loc[x_test.index]

# LightGBM 모델 학습
print("\n" + "=" * 80)
print(f"LightGBM 모델 학습 시작 (특징 개수: {x_train.shape[1]}개, Window Size: {window_size}일)")
print("=" * 80)

# LightGBM Regressor를 MultiOutputRegressor로 래핑하여 5개 타겟 동시 학습
base_estimator = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
multioutput_model = MultiOutputRegressor(base_estimator)

# 학습 실행
multioutput_model.fit(x_train, y_train)

# 예측 실행 (5개 변화율 예측)
y_predict_chg_rate = multioutput_model.predict(x_test)
y_predict_chg_rate_df = pd.DataFrame(y_predict_chg_rate, columns=target_cols, index=x_test.index)


# # 예측된 변화율을 KRW 레벨로 역변환 

# # 예측 레벨 (KRW) 초기화
# predicted_level_df = pd.DataFrame(index=x_test.index)

# # 가격 항목 (O, H, L, C) 역변환: 예측된 % 변화율을 KRW 금액으로 복원
# price_cols = ['open', 'high', 'low', 'close']
# for i, col in enumerate(price_cols):
#     # Predicted Level = Current Level * (1 + Predicted % Change / 100)
#     predicted_level_df[col] = current_level_test[col] * (1 + y_predict_chg_rate_df[f'target_{col}_chg'] / 100)

# # 거래량 항목 (Volume) 역변환: 예측된 Log 변화율을 KRW 금액으로 복원
# # Predicted Volume = Current Volume * (exp(Predicted Log Change) - 1)
# vol_col = 'volume'
# predicted_level_df[vol_col] = current_level_test[vol_col] * (np.expm1(y_predict_chg_rate_df[f'target_{vol_col}_chg']) + 1)


# 항목별 RMSE 평가 (KRW 단위) 

# rmse_results = {}
# report_data = []

# for col in price_cols + [vol_col]:
#     # 실제 레벨 (KRW 또는 Volume)
#     actual_col = f'next_{col}' if col != 'volume' else f'next_{vol_col}'
    
#     # RMSE 계산
#     rmse = np.sqrt(mean_squared_error(next_level_test[actual_col], predicted_level_df[col]))
#     rmse_results[col] = rmse
    
#     # 보고서 데이터 추가
#     if col != 'volume':
#         report_data.append([f"Next {col.upper()} Price (KRW)", f"{rmse:,.2f} KRW"])
#     else:
#         report_data.append([f"Next VOLUME", f"{rmse:,.0f} units"])


# print("\n" + "=" * 80)
# print("LightGBM 테스트셋 결과")
# print("=" * 80)

# headers = ["예측 항목", "RMSE (KRW/Volume 단위)"]
# print("### 각 항목별 예측 오차 (RMSE) ###")
# print(tabulate.tabulate(report_data, headers=headers, tablefmt="markdown"))

# # 주요 항목 시각화 (Close Price)

# # 종가 예측 시각화
# plt.figure(figsize=(16, 8))
# plt.plot(next_level_test.index, next_level_test['next_close'], label='Actual Next Close', color='blue', linewidth=2)
# plt.plot(predicted_level_df.index, predicted_level_df['close'], label=f'Predicted Next Close (RMSE: {rmse_results["close"]:,.0f} KRW)', color='red', linestyle='--', linewidth=1.5)

# plt.title('LightGBM Multi-Output Prediction: Actual vs. Predicted Close Price')
# plt.xlabel('Date')
# plt.ylabel('Price (KRW)')
# plt.legend()
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# 항목별 변화율 RMSE 및 R-squared 평가  
rmse_results = {}
r2_results = {}
report_data = []

for i, col in enumerate(target_cols):
    actual_chg = y_test[col]
    predicted_chg = y_predict_chg_rate_df[col]
    
    # RMSE (변화율 자체의 오차)
    rmse = np.sqrt(mean_squared_error(actual_chg, predicted_chg))
    rmse_results[col] = rmse
    
    # R-squared (변화율 설명력)
    r2 = r2_score(actual_chg, predicted_chg)
    r2_results[col] = r2

    # 보고서 데이터 추가
    display_name = col.replace('target_', '').replace('_chg', '').upper()
    report_data.append([f"Next {display_name}", f"{rmse:.4f}", f"{r2:.6f}"])


print("\n" + "=" * 80)
print("LightGBM (change)")
print("=" * 80)

headers = ["예측 항목", "RMSE (변화율)", "R-squared (변화율)"]
print("### 각 항목별 예측 오차 및 설명력 ###")
print(tabulate.tabulate(report_data, headers=headers, tablefmt="markdown"))

# --- 7. 시각화 (5개 변수 전체 변화율 비교) ---

fig, axes = plt.subplots(5, 1, figsize=(16, 15), sharex=True)

# Volume은 Log Change이므로 별도의 이름 사용
plot_titles = ['Open Change (%)', 'High Change (%)', 'Low Change (%)', 'Close Change (%)', 'Volume Log Change']

for i, col in enumerate(target_cols):
    ax = axes[i]
    
    # 실제 변화율
    ax.plot(y_test.index, y_test[col], label='Actual Change Rate', color='blue', linewidth=1)
    
    # 예측 변화율
    ax.plot(y_predict_chg_rate_df.index, y_predict_chg_rate_df[col], label='Predicted Change Rate', color='red', linestyle='--', linewidth=1)
    
    # 제목 및 지표 표시
    rmse_val = rmse_results[col]
    r2_val = r2_results[col]
    ax.set_title(f"{plot_titles[i]} (RMSE: {rmse_val:.4f}, R^2: {r2_val:.6f})", fontsize=12)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8) # 0% 기준선
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.5)

plt.xlabel('Date')
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # 전체 제목을 위한 여백 조정
plt.show()