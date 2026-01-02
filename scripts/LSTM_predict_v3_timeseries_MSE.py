import os
import datetime
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# =========================
# 0) 설정
# =========================
SEQ_LEN = 20
STRIDE  = 1

time_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 시계열 학습 모델 폴더 (너가 지정)
model_dir = os.path.join('..', 'fit', 'fit_LSTM_timeseries_MSE', '20260102-165535')
save_dir = os.path.join('..', 'results', 'results_LSTM_timeseries_MSE', time_now)
os.makedirs(save_dir, exist_ok=True)

# 모델/스케일러 로드
model = tf.keras.models.load_model(os.path.join(model_dir, 'lstm_timeseries_model.h5'))
scaler_x = joblib.load(os.path.join(model_dir, 'scaler_x.pkl'))
scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))

print("Loaded model:", os.path.join(model_dir, 'lstm_timeseries_model.h5'))
print("Model input_shape:", model.input_shape)

# 테스트 데이터
test_csv  = sorted(glob('../datasets_0102/test/data_*.csv'))
test_json = sorted(glob('../datasets_0102/test/curve_fit_result-joint_angle_*.json'))

if len(test_csv) == 0 or len(test_json) == 0:
    raise FileNotFoundError("No test csv/json found.")
if len(test_csv) != len(test_json):
    print("[WARN] CSV/JSON counts differ. zip pairing may be wrong.")
    print("CSV sample:", test_csv[:3])
    print("JSON sample:", test_json[:3])

# =========================
# 1) Joint Angle 길이 결정
# =========================
def safe_len(x):
    try:
        return len(x)
    except:
        return -1

tmp = pd.read_json(test_json[0])
column_size = int(tmp["Joint Angle"].apply(safe_len).mode().iloc[0])
print("Expected Joint Angle length:", column_size)

# 입력/출력 컬럼
input_columns = ['wire length #0', 'wire length #1', 'loadcell #0', 'loadcell #1'] + \
                [f'Joint Angle_{i}' for i in range(column_size)]
output_columns = ['fx_kalman', 'fy_kalman']


# =========================
# 2) 파일별로 처리: 병합 -> 시퀀스 생성 -> 예측 -> 프레임에 매핑
# =========================
all_pred_rows = []
all_frame_rows = []

for csv_path, json_path in zip(test_csv, test_json):
    df_csv = pd.read_csv(csv_path).reset_index(drop=True)
    df_json = pd.read_json(json_path).reset_index(drop=True)

    n = min(len(df_csv), len(df_json))
    if len(df_csv) != len(df_json):
        print(f"[WARN] length mismatch: {os.path.basename(csv_path)}({len(df_csv)}) vs {os.path.basename(json_path)}({len(df_json)}) -> trunc {n}")

    df_csv = df_csv.iloc[:n].reset_index(drop=True)
    df_json = df_json.iloc[:n].reset_index(drop=True)

    df = pd.concat([df_csv, df_json], axis=1)

    # 깨진 Joint Angle 제거
    ja_len = df["Joint Angle"].apply(safe_len)
    df = df[ja_len == column_size].reset_index(drop=True)

    # Joint Angle 펼치기
    joint_angle = np.array(df["Joint Angle"].tolist(), dtype=np.float32)  # (T, column_size)
    joint_angle_df = pd.DataFrame(joint_angle, columns=[f'Joint Angle_{i}' for i in range(column_size)])
    df = pd.concat([df.reset_index(drop=True), joint_angle_df], axis=1)

    # 필요한 컬럼 NaN 제거
    df = df.dropna(subset=input_columns + output_columns).reset_index(drop=True)

    T = len(df)
    if T < SEQ_LEN:
        print(f"[SKIP] {os.path.basename(csv_path)} (T={T}) < SEQ_LEN={SEQ_LEN}")
        continue

    # 시퀀스 만들기
    X_list = []
    end_indices = []  # 각 시퀀스가 대응하는 "마지막 프레임 인덱스"

    x_raw = df[input_columns].values.astype(np.float32)      # (T, F)
    y_raw = df[output_columns].values.astype(np.float32)     # (T, 2)

    for s in range(0, T - SEQ_LEN + 1, STRIDE):
        e = s + SEQ_LEN
        X_list.append(x_raw[s:e])     # (SEQ_LEN, F)
        end_indices.append(e - 1)     # 마지막 프레임 인덱스

    X = np.array(X_list, dtype=np.float32)  # (Nseq, SEQ_LEN, F)

    # 정규화: (Nseq*SEQ_LEN, F)로 펴서 scaler 적용
    X_flat = X.reshape(-1, X.shape[-1])
    X_norm = scaler_x.transform(X_flat).reshape(X.shape)

    # 예측
    pred_norm = model.predict(X_norm, verbose=0)
    pred = scaler_y.inverse_transform(pred_norm)  # (Nseq, 2)

    # 프레임 단위로 pred 컬럼 만들기 (기본 NaN, 예측 가능한 프레임만 채움)
    df_out = df.copy()
    df_out["pred_fx"] = np.nan
    df_out["pred_fy"] = np.nan

    for k, idx in enumerate(end_indices):
        df_out.at[idx, "pred_fx"] = pred[k, 0]
        df_out.at[idx, "pred_fy"] = pred[k, 1]

        all_pred_rows.append({
            "src_csv": os.path.basename(csv_path),
            "frame_idx": int(idx),
            "gt_fx": float(y_raw[idx, 0]),
            "gt_fy": float(y_raw[idx, 1]),
            "pred_fx": float(pred[k, 0]),
            "pred_fy": float(pred[k, 1]),
        })

    df_out["src_csv"] = os.path.basename(csv_path)
    all_frame_rows.append(df_out)

    # 파일별 저장(원하면)
    per_file_save = os.path.join(save_dir, f"dataframe_with_pred_{os.path.basename(csv_path).replace('.csv','')}.csv")
    df_out.to_csv(per_file_save, index=False)

# =========================
# 3) 전체 저장
# =========================
pred_df = pd.DataFrame(all_pred_rows)
frame_df = pd.concat(all_frame_rows, ignore_index=True) if len(all_frame_rows) else pd.DataFrame()

pred_save = os.path.join(save_dir, f"predicted_timeseries_results_{time_now}.csv")
pred_df.to_csv(pred_save, index=False)

frame_save = os.path.join(save_dir, f"dataframe_with_pred_all_{time_now}.csv")
frame_df.to_csv(frame_save, index=False)

print("Saved:")
print(" -", pred_save)
print(" -", frame_save)
