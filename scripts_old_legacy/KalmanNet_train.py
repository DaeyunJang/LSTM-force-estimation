import pandas as pd
from glob import glob
import numpy as np
import os
import datetime
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# CSV 및 JSON 파일 불러오기
data_csv = sorted(glob('../datasets_0102_offset_30deg/train/data_LPF_2*.csv'))
curvefit_json = sorted(glob('../datasets_0102_offset_30deg/train/curve_fit_result-joint_angle_*.json'))

csv_dataframes = [pd.read_csv(file) for file in data_csv]
json_dataframes = [pd.read_json(file) for file in curvefit_json]

raw_dataframe = pd.concat(csv_dataframes)
curvefit_dataframe = pd.concat(json_dataframes)

data_expanded = pd.concat([raw_dataframe, curvefit_dataframe], axis=1)

# Joint Angle을 개별 열로 변환
column_size = len(data_expanded['Joint Angle'].iloc[0])
joint_angle = np.array(data_expanded['Joint Angle'].tolist())
joint_angle_df = pd.DataFrame(joint_angle, columns=[f'Joint Angle_{i}' for i in range(column_size)])
final_df = pd.concat([data_expanded.reset_index(drop=True), joint_angle_df], axis=1)

# 입력 및 출력 컬럼 정의
input_columns = ['wire length #0', 'wire length #1', 'loadcell #0', 'loadcell #1'] + [f'Joint Angle_{i}' for i in range(column_size)]
output_columns = ['fx_kalman', 'fy_kalman']

x = final_df[input_columns].values
y = final_df[output_columns].values

# 데이터 정규화
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_normalized = scaler_x.fit_transform(x)
y_normalized = scaler_y.fit_transform(y)

# 학습/검증 데이터 분할
x_train, x_valid, y_train, y_valid = train_test_split(x_normalized, y_normalized, test_size=0.2, random_state=42)

# 스케일러 저장
save_dir = '../fit'
save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(save_dir, exist_ok=True)

joblib.dump(scaler_x, os.path.join(save_dir, 'scaler_x.pkl'))
joblib.dump(scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))

# ✅ **KalmanNet 구현 (TensorFlow 기반)**
class KalmanNet(Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KalmanNet, self).__init__()
        self.dense1 = Dense(hidden_dim, activation='relu')
        self.dense2 = Dense(hidden_dim, activation='relu')
        self.kalman_gain_layer = Dense(output_dim, activation='sigmoid')  # Kalman Gain 학습 부분
        self.output_layer = Dense(output_dim)

    def call(self, x, prev_state):
        h = self.dense1(x)
        h = self.dense2(h)
        kalman_gain = self.kalman_gain_layer(h)  # Kalman Gain을 0~1 범위로 제한
        y_pred = self.output_layer(h)

        # Kalman 필터의 업데이트 공식 적용
        innovation = y_pred - prev_state  # 예측값과 이전 상태 차이
        updated_state = prev_state + kalman_gain * innovation

        return updated_state

# 모델 초기화
input_dim = x_train.shape[1]
hidden_dim = 128
output_dim = y_train.shape[1]

kalman_model = KalmanNet(input_dim, hidden_dim, output_dim)
kalman_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# ✅ **데이터 형태 조정 (LSTM처럼 3D로 변환할 필요 없음)**
x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)

# ✅ **KalmanNet 학습**
epochs = 100
batch_size = 128

# 초기 상태 설정 (0으로 초기화)
prev_state_train = np.zeros((x_train.shape[0], output_dim))
prev_state_valid = np.zeros((x_valid.shape[0], output_dim))

history = kalman_model.fit(
    x_train, prev_state_train,
    epochs=epochs, batch_size=batch_size,
    validation_data=(x_valid, prev_state_valid),
    verbose=1
)

# ✅ **모델 평가**
loss, mae = kalman_model.evaluate(x_valid, prev_state_valid, verbose=1)
print(f'Validation Loss: {loss}, Validation MAE: {mae}')

# 모델 저장
kalman_model.save(os.path.join(save_dir, 'kalman_net.h5'))
print("KalmanNet 모델 저장 완료!")

# ✅ **예측**
predicted = kalman_model.predict(x_valid)

# 결과 역정규화
predicted_original = scaler_y.inverse_transform(predicted)
y_valid_original = scaler_y.inverse_transform(y_valid)

# 예측값 및 실제값 저장
np.savetxt(os.path.join(save_dir, 'predicted_original.csv'), predicted_original, delimiter=',')
np.savetxt(os.path.join(save_dir, 'y_val_original.csv'), y_valid_original, delimiter=',')

print("예측값:", predicted_original)
print("실제값:", y_valid_original)
