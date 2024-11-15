import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
# from keras.src.backend.jax.random import dropout

# 여러 개의 CSV와 JSON 파일 경로를 지정합니다.
# data_csv = sorted(glob('../datasets/train/data_LPF_*.csv'))
data_csv = sorted(glob('../datasets_0.4mm_legacy/train/data_LPF_2*.csv'))
curvefit_json = sorted(glob('../datasets_0.4mm_legacy/train/curve_fit_result-joint_angle_*.json'))

# 모든 CSV 파일을 읽어 리스트에 저장합니다.
csv_dataframes = [pd.read_csv(file) for file in data_csv]

# 모든 JSON 파일을 읽어 리스트에 저장합니다.
json_dataframes = [pd.read_json(file) for file in curvefit_json]

raw_dataframe = pd.concat([df for df in csv_dataframes])
curvefit_dataframe = pd.concat([df for df in json_dataframes])

# 병합된 CSV와 JSON 데이터를 하나의 데이터프레임으로 병합합니다.
data_expanded = pd.concat([raw_dataframe, curvefit_dataframe], axis=1)

# Curve fitting에서 Joint Angle 배열을 분리
column_size = len(data_expanded['Joint Angle'].iloc[0])
# Joint Angle 배열을 개별 열로 변환
joint_angle = np.array(data_expanded['Joint Angle'].tolist())
joint_angle_df = pd.DataFrame(joint_angle, columns=[f'Joint Angle_{i}' for i in range(column_size)])
# 기존 데이터프레임과 Joint Angle 개별 열을 합침
final_df = pd.concat([data_expanded.reset_index(drop=True), joint_angle_df], axis=1)

# 병합된 데이터를 확인합니다.
print(final_df.head())

# 데이터가 준비되었으므로, 이제 모델을 학습할 수 있습니다.
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os, sys

print(tf.config.list_physical_devices('GPU'))

save_dir = '../fit'
save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 입력과 출력 데이터 분리
# joint angles are not normalized before (from `save_image_curvefit_params_csv_json.py`)
# normalized data
input_columns = ['wire length #0', 'wire length #1', 'loadcell #0', 'loadcell #1']
# non-normalized data
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

x_normalized = np.concatenate([x_non_joint_angle_normalized, x_joint_angle], axis=1)
# 학습/검증 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(x_normalized, y_normalized, test_size=0.2, random_state=42)

# 스케일러 저장
joblib.dump(scaler_x, os.path.join(save_dir, 'scaler_x.pkl'))
joblib.dump(scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))

# 입력 데이터 차원 조정 (LSTM에 맞게)
x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)

# TensorBoard 설정
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=save_dir, histogram_freq=1)

# 조기 종료 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#
# # LSTM 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(x_train.shape[1], 1)),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),  # 정규화 추가
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # 정규화 추가
    # tf.keras.layers.Dense(32, activation='relu'),  # 정규화 추가
    tf.keras.layers.Dense(2)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])

# 모델 학습
history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_val, y_val), callbacks=[tensorboard_callback], verbose=1)

# 모델 평가
loss, mae = model.evaluate(x_val, y_val, verbose=1)
print(f'Validation Loss: {loss}, Validation MAE: {mae}')

# 모델 저장
model.save(os.path.join(save_dir, 'lstm_model.h5'))

# 예측
predicted = model.predict(x_val)

# 결과 역정규화
predicted_original = scaler_y.inverse_transform(predicted)
y_val_original = scaler_y.inverse_transform(y_val)

# 예측값 및 실제값 저장
np.savetxt(save_dir + '/predicted_original.csv', predicted_original, delimiter=',')
np.savetxt(save_dir + '/y_val_original.csv', y_val_original, delimiter=',')

print("예측값:", predicted_original)
print("실제값:", y_val_original)

###################################################################################
# 저장된 모델 로드 예제
loaded_model = tf.keras.models.load_model(os.path.join(save_dir, 'lstm_model.h5'))
# loaded_model = tf.keras.models.load_model('../fit/20240919-144705/lstm_model.h5')

# 로드된 모델로 예측
loaded_predicted = loaded_model.predict(x_val)
loaded_predicted_original = scaler_y.inverse_transform(loaded_predicted)
print("로드된 모델의 예측값:", loaded_predicted_original)

###################################################################################
import subprocess
import webbrowser
import time

# TensorBoard 명령어 실행
subprocess.Popen(['tensorboard', '--logdir', save_dir])

# TensorBoard가 시작될 때까지 대기
time.sleep(1)

# 기본 브라우저에서 TensorBoard URL 열기
url = 'http://localhost:6006/'
webbrowser.open(url)
