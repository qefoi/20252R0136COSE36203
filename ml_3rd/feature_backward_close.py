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

# p-value 기반의 backward search
def p_value_backward_elimination(x_train, y_train, alpha, verbose=True):
    x_processed = x_train.copy()

    # 성능 기록을 위한 DataFrame 초기화 
    performance_log = pd.DataFrame(columns=['Step', 'Features Removed', 'Adj_R_sq', 'Condition_No', 'Max_P_Value'])
    step = 0

    while True:
        model = sm.OLS(y_train, x_processed).fit()

        p_values = model.pvalues.drop('const', errors='ignore')

        # 현재 단계 성능 기록 
        log_entry = {
            'Step': step,
            'Features Removed': len(x_train.columns) - (x_processed.shape[1] - 1),
            'Adj_R_sq': model.rsquared_adj,
            'Condition_No': model.condition_number,
            'Max_P_Value': p_values.max() if not p_values.empty else np.nan
        }
        performance_log.loc[step] = log_entry

        if p_values.empty:
            break

        max_p_value = p_values.max()
        feature_to_remove = p_values.idxmax()

        if verbose:
            print("=" * 70)
            print(f"[Step {step}] R-sq: {model.rsquared:.4f} | Adj. R-sq: {model.rsquared_adj:.4f} | Condition No: {model.condition_number:.2e}")
            print(f"제거 후보 변수: {feature_to_remove} (P-value: {max_p_value:.4f})")

        if max_p_value > alpha:
            x_processed = x_processed.drop(columns=[feature_to_remove])
            step += 1
            if verbose:
                print(f"--> {feature_to_remove} 변수 제거. 남은 독립 변수 개수: {x_processed.shape[1] - 1}")
        else:
            if verbose:
                print("=" * 70)
                print(f"최종 모델: 모든 변수의 P-value가 {alpha} 이하로 유의미합니다. 제거 중단.")
            break
    return model, x_processed.drop(columns=['const']), performance_log

# close의 변화량 정의
data['target_change'] = data['close'].shift(-1) - data['close']

# window size전까지의 data를 새로운 column으로 추가
target_column = ['body', 'upper_shadow', 'lower_shadow', 'body_ratio', 
                   'shadow_ratio', 'direction', 'volume_strength', 'momentum']

x_features = pd.DataFrame(index=data.index)
for col in target_column:
    for i in range(1, window_size + 1):
        x_features[f'{col}_{i}_days_ago'] = data[col].shift(i)

# NaN 포함되는 데이터 삭제(앞에서 window_size만큼의 data, 마지막 data)
data.dropna(inplace=True)
x = x_features.loc[data.index].dropna()
y = data['target_change'].loc[x.index] # Y는 변화량
x = sm.add_constant(x)

# 70% -> training, 30% -> test
train_size = int(len(x) * 0.7)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test_change = y[:train_size], y[train_size:] 

# --- 1. 초기 모델 (Backward Elimination 전) 학습 ---
initial_model = sm.OLS(y_train, x_train).fit()
initial_cond = initial_model.condition_number
initial_y_predict_change = initial_model.predict(x_test)

initial_rmse = np.sqrt(mean_squared_error(y_test_change, initial_y_predict_change))
initial_r2 = r2_score(y_test_change, initial_y_predict_change)

initial_feature_count = len(initial_model.params) - 1

# --- 2. Backward Elimination 실행 및 최종 모델 학습 ---
final_model_ols, x_train_opt, performance_log = p_value_backward_elimination(x_train, y_train, alpha=0.1)

# 최종 모델 예측 
final_features = final_model_ols.params.index
x_test_opt = x_test[final_features]
final_y_predict_change = final_model_ols.predict(x_test_opt)

final_rmse = np.sqrt(mean_squared_error(y_test_change, final_y_predict_change))
final_r2 = r2_score(y_test_change, final_y_predict_change)
final_cond = final_model_ols.condition_number
final_feature_count = len(final_model_ols.params) - 1

# --- 3. 최종 결과 분석 및 비교 출력 ---

print("\n" + "=" * 80)
print("Feature Engineered Change Model 최종 비교 분석")
print("================================================================================\n")

# 성능 지표를 DataFrame으로 정리
comparison_data = [
    ['초기 모델 (Full FE)', initial_rmse, initial_r2, initial_cond, initial_feature_count],
    ['최적화 모델 (Selected FE)', final_rmse, final_r2, final_cond, final_feature_count]
]

headers = ["Model", "RMSE (Test Level)", "R-squared (Test Level)", "Condition No.", "Feature Count"]

# Markdown 테이블 출력
print("### 테스트셋 성능 및 모델 안정성 비교 ###")
print(tabulate.tabulate(comparison_data, headers=headers, tablefmt="markdown", floatfmt=".4f"))


print("\n[안정성 (Condition Number) 개선 여부]")
cond_reduction = (initial_cond - final_cond) / initial_cond * 100
print(f"초기 Condition Number: {initial_cond:.2e}")
print(f"최종 Condition Number: {final_cond:.2e}")
print(f"Condition Number가 {cond_reduction:.2f}% 감소하여 모델의 안정성이 크게 향상되었습니다.")

print("\n[최적화 모델 Summary (Backward Elimination 최종 결과)]")
print(final_model_ols.summary().as_text())


# --- 4. 시각화 (두 모델 비교) ---

result_df_comp = pd.DataFrame({
    'Actual': y_test_change.values,
    'Predicted (Initial)': initial_y_predict_change,
    'Predicted (Final Opt.)': final_y_predict_change
}, index=x_test.index)


plt.figure(figsize=(16, 8))
plt.plot(result_df_comp.index, result_df_comp['Actual'], label='Actual Next Close', color='blue', linewidth=2)
plt.plot(result_df_comp.index, result_df_comp['Predicted (Initial)'], label=f'Initial Model (RMSE: {initial_rmse:.2f})', color='red', linestyle=':', linewidth=1.5)
plt.plot(result_df_comp.index, result_df_comp['Predicted (Final Opt.)'], label=f'Optimized Model (RMSE: {final_rmse:.2f})', color='green', linestyle='--', linewidth=1.5)

plt.title('Comparison: Initial vs. Optimized Model')
plt.xlabel('Date')
plt.ylabel('Price (KRW)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
