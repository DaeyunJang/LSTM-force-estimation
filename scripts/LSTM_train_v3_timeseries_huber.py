import os, sys
import datetime
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# 0) 시계열 설정 + pretrain 토글
# =========================
SEQ_LEN = 20
STRIDE  = 1
EPOCHS = 200
BATCH_SIZE = 512
USE_PRETRAIN = False   # True/False 토글
PRETRAIN_PATH = os.path.join("..", "model", "lstm_model.h5")  # 초기 가중치로 쓸 h5 경로(시계열 모델이어야 함)

# 여러 개의 CSV와 JSON 파일 경로
data_csv = sorted(glob('../datasets_0102/train/data_*.csv'))
joint_angle_json = sorted(glob('../datasets_0102/train/curve_fit_result-joint_angle_*.json'))

if len(data_csv) == 0:
    raise FileNotFoundError("No CSV files found.")
if len(joint_angle_json) == 0:
    raise FileNotFoundError("No JSON files found.")

print("#CSV:", len(data_csv), " #JSON:", len(joint_angle_json))
if len(data_csv) != len(joint_angle_json):
    print("[WARN] CSV/JSON counts differ. zip pairing may be wrong.")
    print("CSV sample:", data_csv[:3])
    print("JSON sample:", joint_angle_json[:3])

print(tf.config.list_physical_devices('GPU'))

# 저장 폴더
save_dir = os.path.join('..', 'fit', 'fit_LSTM_timeseries_huber')
save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(save_dir, exist_ok=True)


# =========================
# 1) Joint Angle length 결정(대부분 9)
# =========================
def safe_len(x):
    try:
        return len(x)
    except:
        return -1

tmp = pd.read_json(joint_angle_json[0])
col_size = int(tmp["Joint Angle"].apply(safe_len).mode().iloc[0])
print("Expected Joint Angle length:", col_size)


# =========================
# 2) input/output columns
# =========================
input_columns = ['wire length #0', 'wire length #1', 'loadcell #0', 'loadcell #1'] + \
                [f'Joint Angle_{i}' for i in range(col_size)]
output_columns = ['fx_kalman', 'fy_kalman']


# =========================
# 3) 파일별로 병합 -> Joint Angle 펼치기 -> 시퀀스 생성
# =========================
X_list = []
y_list = []

for csv_path, json_path in zip(data_csv, joint_angle_json):
    df_csv = pd.read_csv(csv_path).reset_index(drop=True)
    df_json = pd.read_json(json_path).reset_index(drop=True)

    n = min(len(df_csv), len(df_json))
    if len(df_csv) != len(df_json):
        print(f"[WARN] length mismatch: {os.path.basename(csv_path)}({len(df_csv)}) vs {os.path.basename(json_path)}({len(df_json)}) -> trunc {n}")

    df_csv = df_csv.iloc[:n].reset_index(drop=True)
    df_json = df_json.iloc[:n].reset_index(drop=True)

    data_expanded = pd.concat([df_csv, df_json], axis=1)

    # 깨진 joint angle row 제거
    ja_len = data_expanded["Joint Angle"].apply(safe_len)
    data_expanded = data_expanded[ja_len == col_size].reset_index(drop=True)

    # 펼치기
    joint_angle = np.array(data_expanded['Joint Angle'].tolist(), dtype=np.float32)  # (T, col_size)
    joint_angle_df = pd.DataFrame(joint_angle, columns=[f'Joint Angle_{i}' for i in range(col_size)])
    final_df = pd.concat([data_expanded.reset_index(drop=True), joint_angle_df], axis=1)

    # 필요한 컬럼 NaN 제거
    final_df = final_df.dropna(subset=input_columns + output_columns).reset_index(drop=True)

    x_raw = final_df[input_columns].values.astype(np.float32)   # (T, 13)
    y_raw = final_df[output_columns].values.astype(np.float32)  # (T, 2)

    T = len(final_df)
    if T < SEQ_LEN:
        continue

    for s in range(0, T - SEQ_LEN + 1, STRIDE):
        e = s + SEQ_LEN
        X_list.append(x_raw[s:e])      # (SEQ_LEN, 13)
        y_list.append(y_raw[e-1])      # 마지막 프레임을 label로

X = np.array(X_list, dtype=np.float32)   # (N, SEQ_LEN, 13)
y = np.array(y_list, dtype=np.float32)   # (N, 2)

print("X:", X.shape, "y:", y.shape)
if X.shape[0] == 0:
    raise RuntimeError("No sequences built. Reduce SEQ_LEN or check dataset lengths.")


# =========================
# 4) Train/Valid split
# =========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Train:", X_train.shape, y_train.shape)
print("Valid:", X_valid.shape, y_valid.shape)


# =========================
# 5) 정규화 (train 기준)
#    - 시계열이라 (N, T, F) -> (N*T, F)로 펴서 scaler fit/transform
# =========================
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_flat = X_train.reshape(-1, X_train.shape[-1])  # (N*T, F)
X_valid_flat = X_valid.reshape(-1, X_valid.shape[-1])

scaler_x.fit(X_train_flat)
X_train_n = scaler_x.transform(X_train_flat).reshape(X_train.shape)
X_valid_n = scaler_x.transform(X_valid_flat).reshape(X_valid.shape)

scaler_y.fit(y_train)
y_train_n = scaler_y.transform(y_train)
y_valid_n = scaler_y.transform(y_valid)

joblib.dump(scaler_x, os.path.join(save_dir, 'scaler_x.pkl'))
joblib.dump(scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))


# =========================
# 6) 모델 (시계열 입력: (SEQ_LEN, 13))
#    - pretrain on/off
# =========================
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=save_dir, histogram_freq=1)

num_features = X_train_n.shape[-1]

def build_model(seq_len, n_feat):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_feat)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(32, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(2)
    ])
    return model

if USE_PRETRAIN:
    if not os.path.exists(PRETRAIN_PATH):
        raise FileNotFoundError(f"PRETRAIN_PATH not found: {PRETRAIN_PATH}")
    model = tf.keras.models.load_model(PRETRAIN_PATH)
    print("[INFO] Loaded pretrained model:", PRETRAIN_PATH)
    # 입력 shape 호환 체크
    print("Loaded model input_shape:", model.input_shape, "expected:", (None, SEQ_LEN, num_features))
else:
    model = build_model(SEQ_LEN, num_features)

model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=['mae']
)

model.summary()


# =========================
# 7) 학습
# =========================
history = model.fit(
    X_train_n, y_train_n,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_valid_n, y_valid_n),
    callbacks=[tensorboard_callback, early_stopping, reduce_lr],
    verbose=1
)

loss, mae = model.evaluate(X_valid_n, y_valid_n, verbose=1)
print(f'Validation Loss: {loss}, Validation MAE: {mae}')


# =========================
# 8) 저장 + valid 예측 저장
# =========================
model.save(os.path.join(save_dir, 'lstm_timeseries_model.h5'))

pred_n = model.predict(X_valid_n)
pred = scaler_y.inverse_transform(pred_n)
gt   = scaler_y.inverse_transform(y_valid_n)

np.savetxt(os.path.join(save_dir, 'predicted_value.csv'), pred, delimiter=',')
np.savetxt(os.path.join(save_dir, 'original_value.csv'), gt, delimiter=',')

# gt + pred 합쳐서 저장 (요청)
gt_pred = np.hstack([gt, pred])  # [gt_fx, gt_fy, pred_fx, pred_fy]
np.savetxt(os.path.join(save_dir, 'gt_and_pred.csv'), gt_pred, delimiter=',',
           header='gt_fx,gt_fy,pred_fx,pred_fy', comments='')

print("Saved:", save_dir)


# =========================
# 9) TensorBoard 실행 (옵션)
# =========================
import subprocess
import webbrowser
import time

subprocess.Popen(['tensorboard', '--logdir', save_dir])
time.sleep(1)
webbrowser.open('http://localhost:6006/')
