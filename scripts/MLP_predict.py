import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from glob import glob
import datetime
import os

time_now =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = os.path.join('..', 'fit', 'fit_MLP', '20241222-215311')
save_dir = os.path.join('..', 'results', 'results_MLP')
save_dir = os.path.join(save_dir, time_now)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 모델 불러오기
model = tf.keras.models.load_model(os.path.join(model_dir, 'MLP_model.h5'))

# 스케일러 불러오기
scaler_x = joblib.load(os.path.join(model_dir, 'scaler_x.pkl'))
scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))

# 여러 개의 테스트용 CSV와 JSON 파일 경로를 지정합니다.
test_csv = sorted(glob('../datasets_20241206-realworld/test/data_2*.csv'))
test_json = sorted(glob('../datasets_20241206-realworld/test/curve_fit_result-joint_angle_*.json'))
# 모든 CSV 파일을 읽어 리스트에 저장합니다.
csv_test_dataframes = [pd.read_csv(file) for file in test_csv]

# 모든 JSON 파일을 읽어 리스트에 저장합니다.
json_test_dataframes = [pd.read_json(file) for file in test_json]

# 병합된 테스트용 CSV와 JSON 데이터를 하나의 데이터프레임으로 병합합니다.
test_raw_dataframe = pd.concat([df for df in csv_test_dataframes])
test_curvefit_dataframe = pd.concat([df for df in json_test_dataframes])
test_data_expanded = pd.concat([test_raw_dataframe, test_curvefit_dataframe], axis=1)

#############################################################################
# # 각 리스트의 길이 계산
# lengths = [len(x) for x in test_data_expanded['Joint Angles']]
#
# # 기준 길이 (예: 가장 많이 등장하는 길이)
# from collections import Counter
# most_common_length = Counter(lengths).most_common(1)[0][0]
#
# # 길이가 다른 항목의 인덱스 찾기
# different_indices = [i for i, length in enumerate(lengths) if length != most_common_length]
#
# print("길이가 다른 항목의 인덱스:", different_indices)
# print("해당 인덱스의 길이:", [lengths[i] for i in different_indices])
#############################################################################

# Curve fitting에서 Joint Angle 배열을 분리
column_size = len(test_data_expanded['Joint Angles'].iloc[0])
# Joint Angle 배열을 개별 열로 변환
joint_angle = np.array(test_data_expanded['Joint Angles'].tolist())
joint_angle_df = pd.DataFrame(joint_angle, columns=[f'Joint Angles_{i}' for i in range(column_size)])
# 기존 데이터프레임과 Joint Angle 개별 열을 합침
final_test_df = pd.concat([test_data_expanded.reset_index(drop=True), joint_angle_df], axis=1)
final_test_df.to_csv('out.csv', index=False)

# 입력 데이터 분리
input_columns = ['wire length #0', 'wire length #1', 'loadcell #0', 'loadcell #1'] + [f'Joint Angles_{i}' for i in range(column_size)]
output_columns = ['fx', 'fy']

x = final_test_df[input_columns].values
x_normalized = scaler_x.transform(x)
x_test = np.expand_dims(x_normalized, axis=-1)


# 예측
predicted_normalized = model.predict(x_test)

# 예측값 역정규화
predicted_original = scaler_y.inverse_transform(predicted_normalized)

# 예측 결과를 데이터프레임으로 변환
# predicted_df = pd.DataFrame(predicted_original, columns=['fx', 'fy'])
results_df = final_test_df[output_columns].copy()
results_df[['pred_fx', 'pred_fy']] = predicted_original

# 결과를 CSV 파일로 저장
time_now =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_file = os.path.join(save_dir, 'predicted_results_with_original_' + time_now + '.csv')
results_df.to_csv(save_file, index=False)
final_test_df.to_csv(os.path.join(save_dir, 'dataframe_' + time_now + '.csv'), index=False)

print(f"예측값이 {save_file} 파일에 저장되었습니다.")