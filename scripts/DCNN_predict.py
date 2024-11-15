import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from glob import glob
import datetime
import os

model_dir = os.path.join('..', 'fit_DCNN', '20241009-134855')

save_dir = '../results_DCNN'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 모델 불러오기 또는 새 모델 정의
if os.path.exists(os.path.join(model_dir, 'dcnn_model.h5')):
    model = tf.keras.models.load_model(os.path.join(model_dir, 'dcnn_model.h5'), compile=False)
else:
    print(f'check the model file (.h5)')

# 스케일러 불러오기
scaler_x = joblib.load(os.path.join(model_dir, 'scaler_x.pkl'))
scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))

# 여러 개의 테스트용 CSV와 JSON 파일 경로를 지정합니다.
test_csv = sorted(glob('../datasets_0.4mm_legacy/test/data_LPF_2*.csv'))
test_json = sorted(glob('../datasets_0.4mm_legacy/test/curve_fit_result-joint_angle_*.json'))

# 모든 CSV 파일을 읽어 리스트에 저장합니다.
csv_test_dataframes = [pd.read_csv(file) for file in test_csv]

# 모든 JSON 파일을 읽어 리스트에 저장합니다.
json_test_dataframes = [pd.read_json(file) for file in test_json]

# 병합된 테스트용 CSV와 JSON 데이터를 하나의 데이터프레임으로 병합합니다.
test_raw_dataframe = pd.concat([df for df in csv_test_dataframes])
test_curvefit_dataframe = pd.concat([df for df in json_test_dataframes])
test_data_expanded = pd.concat([test_raw_dataframe, test_curvefit_dataframe], axis=1)

# Joint Angle 데이터를 개별 열로 분리
coeff_size = len(test_data_expanded['Joint Angle'].iloc[0])
coefficients = np.array(test_data_expanded['Joint Angle'].tolist())
coefficients_df = pd.DataFrame(coefficients, columns=[f'Joint Angle_{i}' for i in range(coeff_size)])

# 기존 데이터프레임과 Joint Angle 개별 열을 합침
final_test_df = pd.concat([test_data_expanded.reset_index(drop=True), coefficients_df], axis=1)

# 입력 데이터 분리
input_non_joint_angle_columns = ['wire length #0', 'wire length #1', 'loadcell #0', 'loadcell #1']
input_joint_angle_columns = [f'Joint Angle_{i}' for i in range(coeff_size)]
output_columns = ['fx', 'fy']

x_test_non_joint_angle = final_test_df[input_non_joint_angle_columns].values
x_test_joint_angle = final_test_df[input_joint_angle_columns].values

# 입력 데이터 스케일링
x_test_non_joint_angle_normalized = scaler_x.transform(x_test_non_joint_angle)
x_test_normalized = np.concatenate([x_test_non_joint_angle_normalized, x_test_joint_angle], axis=1)

# 입력 데이터 차원 조정 (모델 입력 형태에 맞게)
x_test = np.expand_dims(x_test_normalized, axis=-1)  # 1D CNN에 맞는 입력 형태로 변환

# 예측
predicted_normalized = model.predict(x_test)

# 예측값 역정규화
predicted_original = scaler_y.inverse_transform(predicted_normalized)

# 예측 결과를 데이터프레임으로 변환
results_df = final_test_df[output_columns].copy()
results_df[['pred_fx', 'pred_fy']] = predicted_original

# 결과를 CSV 파일로 저장
save_dir = '../results_DCNN/predicted_results_with_original_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
results_df.to_csv(save_dir, index=False)

print("예측값이 'predicted_results.csv' 파일에 저장되었습니다.")
