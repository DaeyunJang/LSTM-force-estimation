import os, sys
import json
import platform
import datetime
import subprocess
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
data_csv         = sorted(glob('../datasets_offset/train/data_*.csv'))
joint_angle_json = sorted(glob('../datasets_offset/train/curve_fit_result-joint_angle_*.json'))

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
save_dir = os.path.join('..', 'fit', 'fit_LSTM_timeseries_MSE_offset')
save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(save_dir, exist_ok=True)


# =========================
# META 저장 유틸
# =========================
def _run_cmd(cmd, cwd=None):
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"[ERR] {e}"

def get_git_info(repo_root=".."):
    # repo_root는 LSTM_train.py 기준 상위 폴더가 git repo일 확률이 높아서 기본 ".."
    info = {}
    info["commit"] = _run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
    info["branch"] = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    info["status_short"] = _run_cmd(["git", "status", "--porcelain"], cwd=repo_root)
    info["remote"] = _run_cmd(["git", "remote", "-v"], cwd=repo_root)
    return info

def get_env_info():
    env = {}
    env["python_version"] = sys.version
    env["platform"] = platform.platform()
    env["tensorflow_version"] = tf.__version__
    env["tf_built_with_cuda"] = bool(tf.test.is_built_with_cuda())
    env["tf_built_with_rocm"] = bool(getattr(tf.test, "is_built_with_rocm", lambda: False)())
    env["physical_gpus"] = [d.name for d in tf.config.list_physical_devices('GPU')]
    env["physical_cpus"] = [d.name for d in tf.config.list_physical_devices('CPU')]
    # 드라이버/쿠다 버전은 tf.sysconfig.get_build_info()에 일부 들어갈 때가 있음
    try:
        env["tf_build_info"] = tf.sysconfig.get_build_info()
    except Exception:
        env["tf_build_info"] = {}
    return env

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_model_summary(model, path_txt):
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def save_metadata(meta_dict):
    save_json(os.path.join(save_dir, "meta.json"), meta_dict)

def save_history(history_obj):
    # history.history = dict(list per epoch)
    save_json(os.path.join(save_dir, "history.json"), history_obj.history)


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

length_mismatch_logs = []

for csv_path, json_path in zip(data_csv, joint_angle_json):
    df_csv = pd.read_csv(csv_path).reset_index(drop=True)
    df_json = pd.read_json(json_path).reset_index(drop=True)

    n = min(len(df_csv), len(df_json))
    if len(df_csv) != len(df_json):
        msg = f"{os.path.basename(csv_path)}({len(df_csv)}) vs {os.path.basename(json_path)}({len(df_json)}) -> trunc {n}"
        print("[WARN] length mismatch:", msg)
        length_mismatch_logs.append(msg)

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

    x_raw = final_df[input_columns].values.astype(np.float32)   # (T, F)
    y_raw = final_df[output_columns].values.astype(np.float32)  # (T, 2)

    T = len(final_df)
    if T < SEQ_LEN:
        continue

    for s in range(0, T - SEQ_LEN + 1, STRIDE):
        e = s + SEQ_LEN
        X_list.append(x_raw[s:e])      # (SEQ_LEN, F)
        y_list.append(y_raw[e-1])      # 마지막 프레임을 label로

X = np.array(X_list, dtype=np.float32)   # (N, SEQ_LEN, F)
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
# 6) 모델 (시계열 입력: (SEQ_LEN, F))
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
    print("Loaded model input_shape:", model.input_shape, "expected:", (None, SEQ_LEN, num_features))
else:
    model = build_model(SEQ_LEN, num_features)


# ====== compile 설정(여기 값도 meta에 저장됨) ======
LEARNING_RATE = 0.001
LOSS_NAME = "mse"  # "mse" or "huber"
HUBER_DELTA = 1.0

if LOSS_NAME.lower() == "huber":
    loss_fn = tf.keras.losses.Huber(delta=HUBER_DELTA)
else:
    loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=loss_fn,
    metrics=['mae']
)

# summary 텍스트 저장
save_model_summary(model, os.path.join(save_dir, "model_summary.txt"))
model.summary()


# =========================
# (중요) META 1차 저장: 학습 시작 전에 남길 것들
# =========================
meta = {}
meta["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
meta["save_dir"] = os.path.abspath(save_dir)

# 하이퍼파라미터
meta["hparams"] = {
    "SEQ_LEN": SEQ_LEN,
    "STRIDE": STRIDE,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "USE_PRETRAIN": USE_PRETRAIN,
    "PRETRAIN_PATH": PRETRAIN_PATH,
    "LEARNING_RATE": LEARNING_RATE,
    "LOSS_NAME": LOSS_NAME,
    "HUBER_DELTA": HUBER_DELTA,
    "early_stopping": {"monitor": "val_loss", "patience": 20, "restore_best_weights": True},
    "reduce_lr": {"monitor": "val_loss", "factor": 0.5, "patience": 6, "min_lr": 1e-6},
}

# 데이터 정보
meta["data"] = {
    "data_csv_glob": "../datasets_offset/train/data_*.csv",
    "joint_angle_json_glob": "../datasets_offset/train/curve_fit_result-joint_angle_*.json",
    "num_csv": len(data_csv),
    "num_json": len(joint_angle_json),
    "pairing_by_zip": True,
    "length_mismatch_logs": length_mismatch_logs[:50],  # 너무 길어지면 잘라서 저장
    "input_columns": input_columns,
    "output_columns": output_columns,
    "joint_angle_len_expected": col_size,
}

# 텐서 shape
meta["shapes"] = {
    "X": list(X.shape),
    "y": list(y.shape),
    "X_train": list(X_train.shape),
    "X_valid": list(X_valid.shape),
    "y_train": list(y_train.shape),
    "y_valid": list(y_valid.shape),
}

# 스케일러 정보(스케일러 파일도 저장하니, 여기엔 타입/범위만)
meta["scaler"] = {
    "scaler_x": "MinMaxScaler fitted on X_train_flat",
    "scaler_y": "MinMaxScaler fitted on y_train",
}

# 모델 정보
meta["model"] = {
    "input_shape_expected": [None, SEQ_LEN, int(num_features)],
    "compiled_metrics": ["mae"],
}

# 환경/깃
meta["env"] = get_env_info()
meta["git"] = get_git_info(repo_root="..")

save_metadata(meta)


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

# history 저장
save_history(history)

loss, mae = model.evaluate(X_valid_n, y_valid_n, verbose=1)
print(f'Validation Loss: {loss}, Validation MAE: {mae}')


# =========================
# (중요) META 2차 업데이트: 학습 결과 요약 저장
# =========================
meta["result"] = {
    "final_val_loss": float(loss),
    "final_val_mae": float(mae),
    "best_val_loss": float(np.min(history.history.get("val_loss", [np.nan]))),
    "best_epoch_by_val_loss": int(np.argmin(history.history.get("val_loss", [np.inf]))) if "val_loss" in history.history else None,
}
save_metadata(meta)


# =========================
# 8) 저장 + valid 예측 저장
# =========================
model.save(os.path.join(save_dir, 'lstm_timeseries_model.h5'))

pred_n = model.predict(X_valid_n)
pred = scaler_y.inverse_transform(pred_n)
gt   = scaler_y.inverse_transform(y_valid_n)

np.savetxt(os.path.join(save_dir, 'predicted_value.csv'), pred, delimiter=',')
np.savetxt(os.path.join(save_dir, 'original_value.csv'), gt, delimiter=',')

# gt + pred 합쳐서 저장
gt_pred = np.hstack([gt, pred])  # [gt_fx, gt_fy, pred_fx, pred_fy]
np.savetxt(os.path.join(save_dir, 'gt_and_pred.csv'), gt_pred, delimiter=',',
           header='gt_fx,gt_fy,pred_fx,pred_fy', comments='')

print("Saved:", save_dir)


# =========================
# 9) TensorBoard 실행 (옵션)
# =========================
import time
import webbrowser

subprocess.Popen(['tensorboard', '--logdir', save_dir])
time.sleep(1)
webbrowser.open('http://localhost:6006/')
