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

# p-value 기반의 backward search
def p_value_backward_elimination(x_train, y_train, alpha = 0.05, verbose=True):
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

OHLCV = stock.get_market_ohlcv_by_date(start_date, end_date, ticker_code)
data = OHLCV[['시가', '고가', '저가', '종가', '거래량']]
data.columns = ['open', 'high', 'low', 'close', 'volume']

# 오늘의 데이터를 이용해 다음날의 종가를 예측해야하므로 하루씩 뒤로 밀기
data['next_close'] = data['close'].shift(-1)

# window size전까지의 data를 새로운 column으로 추가
target_column = ['open', 'high', 'low', 'close', 'volume']
for col in target_column:
    for i in range(1, window_size + 1):
        data[f'{col}_{i}_days_ago'] = data[col].shift(i)

# NaN 포함되는 데이터 삭제(앞에서 window_size만큼의 data, 마지막 data)
data.dropna(inplace=True)

x = data.filter(regex='_days_ago$')
y = data['next_close']
x = sm.add_constant(x)
print(f"x 변수 개수 (상수항 포함): {x.shape[1]}")

# 70% -> training, 30% -> test
train_size = int(len(x) * 0.7)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:] 

# backward search 실행
final_model, x_train_opt, performance_log = p_value_backward_elimination(x_train, y_train, alpha=0.05)

# 초기 모델 학습
initial_model = sm.OLS(y_train, x_train).fit()
initial_y_predict = initial_model.predict(x_test)
initial_rmse = np.sqrt(mean_squared_error(y_test, initial_y_predict))
initial_r2 = r2_score(y_test, initial_y_predict)

# feature selection 이후 모델 학습
final_features = final_model.params.index
x_test_opt = x_test[final_features]
opt_y_predict = final_model.predict(x_test_opt)
opt_rmse = np.sqrt(mean_squared_error(y_test, opt_y_predict))
opt_r2 = r2_score(y_test, opt_y_predict)

# 1. 성능 지표 표로 정리 및 비교

print("\n" + "=" * 80)
print("Feature Selection (Backward Elimination) 결과 비교 분석")
print("=" * 80)

# 성능 지표를 DataFrame으로 정리
comparison_df = pd.DataFrame({
    'Model': ['초기 모델 (Full Features)', '최적화 모델 (Selected Features)'],
    'RMSE (Test Set)': [initial_rmse, opt_rmse],
    'R-squared (Test Set)': [initial_r2, opt_r2],
    'Condition Number (Train)': [initial_model.condition_number, final_model.condition_number],
    'Feature Count': [len(initial_model.params) - 1, len(final_model.params) - 1]
})

print("### 테스트셋 성능 및 모델 안정성 비교 ###")
print(comparison_df.to_markdown(index=False, floatfmt=".4f"))


# 2. 안정성 및 성능 해석

print("\n[안정성 (Condition Number) 개선 여부]")
initial_cond = initial_model.condition_number
final_cond = final_model.condition_number
cond_reduction = (initial_cond - final_cond) / initial_cond * 100

print(f"초기 Condition Number: {initial_cond:.2e}")
print(f"최종 Condition Number: {final_cond:.2e}")
if final_cond < initial_cond:
    print(f" Condition Number가 {cond_reduction:.2f}% 감소하여 모델의 안정성이 크게 향상되었습니다. (다중공선성 완화)")
else:
    print("Condition Number가 증가하거나 변화가 미미합니다. 추가적인 조치가 필요합니다.")

print("\n[예측 성능 (RMSE) 변화]")
if opt_rmse < initial_rmse:
    print(f"최적화 모델의 RMSE가 {initial_rmse - opt_rmse:.4f} 만큼 감소했습니다. (성능 개선)")
elif opt_rmse > initial_rmse:
    print(f"최적화 모델의 RMSE가 {opt_rmse - initial_rmse:.4f} 만큼 증가했습니다. 이는 제거된 변수들이 예측에 필요했음을 의미할 수 있습니다.")
else:
    print("RMSE 변화는 미미합니다. 안정성 개선에 집중합니다.")


# 3. 최적화된 변수 목록 확인

print("\n[최종 선택된 변수 목록]")
print(final_model.params.index.tolist())


# 4. 시각화 (두 모델 비교)

result_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted (Initial)': initial_y_predict,
    'Predicted (Optimized)': opt_y_predict
}, index=x_test.index)

plt.figure(figsize=(16, 8))
plt.plot(result_df.index, result_df['Actual'], label='Actual Next Close', color='blue', linewidth=2)
plt.plot(result_df.index, result_df['Predicted (Initial)'], label=f'Initial Model (RMSE: {initial_rmse:.2f})', color='red', linestyle=':', linewidth=1.5)
plt.plot(result_df.index, result_df['Predicted (Optimized)'], label=f'Optimized Model (RMSE: {opt_rmse:.2f})', color='green', linestyle='--', linewidth=1.5)

plt.title('Comparison: Initial Model vs. Optimized Model (Test Set)')
plt.xlabel('Date')
plt.ylabel('Price (KRW)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()