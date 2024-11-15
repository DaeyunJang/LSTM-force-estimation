import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from glob import glob
import datetime
import os
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import joblib
import os

save_dir = os.path.join('..', 'fit_DCNN')
save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 여러 개의 CSV와 JSON 파일 경로를 지정합니다.
train_csv = sorted(glob('../datasets_0.4mm_legacy/train/data_LPF_2*.csv'))
curvefit_json = sorted(glob('../datasets_0.4mm_legacy/train/curve_fit_result-joint_angle_*.json'))

# 모든 CSV 파일을 읽어 리스트에 저장합니다.
csv_dataframes = [pd.read_csv(file) for file in train_csv]

# 모든 JSON 파일을 읽어 리스트에 저장합니다.
json_dataframes = [pd.read_json(file) for file in curvefit_json]

# 병합된 CSV와 JSON 데이터를 하나의 데이터프레임으로 병합합니다.
raw_dataframe = pd.concat([df for df in csv_dataframes])
curvefit_dataframe = pd.concat([df for df in json_dataframes])
data_expanded = pd.concat([raw_dataframe, curvefit_dataframe], axis=1)

# Curve fitting에서 Joint Angle 배열을 분리
column_size = len(data_expanded['Joint Angle'].iloc[0])
joint_angle = np.array(data_expanded['Joint Angle'].tolist())
joint_angle_df = pd.DataFrame(joint_angle, columns=[f'Joint Angle_{i}' for i in range(column_size)])

# 기존 데이터프레임과 Joint Angle 개별 열을 합침
final_df = pd.concat([data_expanded.reset_index(drop=True), joint_angle_df], axis=1)

# 입력과 출력 데이터 분리
input_columns = ['wire length #0', 'wire length #1', 'loadcell #0', 'loadcell #1']
joint_angle_columns = [f'Joint Angle_{i}' for i in range(column_size)]
output_columns = ['fx', 'fy']

x_non_joint_angle = final_df[input_columns].values
x_joint_angle = final_df[joint_angle_columns].values
y = final_df[output_columns].values

# 데이터 정규화
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_non_joint_angle_normalized = scaler_x.fit_transform(x_non_joint_angle)
y_normalized = scaler_y.fit_transform(y)

# Joint Angle 데이터를 결합하여 최종 입력 데이터 생성
x_normalized = np.concatenate([x_non_joint_angle_normalized, x_joint_angle], axis=1)

# 학습/검증 데이터 분할
x_train, x_valid, y_train, y_valid = train_test_split(x_normalized, y_normalized, test_size=0.2, random_state=42)

# 스케일러 저장
joblib.dump(scaler_x, os.path.join(save_dir, 'scaler_x.pkl'))
joblib.dump(scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))

# 입력 데이터 차원 조정 (CNN에 맞도록 차원 확장)
x_train = np.expand_dims(x_train, axis=-1)  # (샘플 수, 피처 수, 1)
x_valid = np.expand_dims(x_valid, axis=-1)

# 모델 정의
model = Sequential()

# 첫 번째 1D 컨볼루션 층
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=1))

# 두 번째 1D 컨볼루션 층
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=1))

# 세 번째 1D 컨볼루션 층
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=1))

# 평탄화 층
model.add(Flatten())

# 완전 연결층
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

# 출력층
model.add(Dense(2))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_valid, y_valid))

# 학습된 모델 저장
model_save_path = os.path.join(save_dir, 'dcnn_model.h5')
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
