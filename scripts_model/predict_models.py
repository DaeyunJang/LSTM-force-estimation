import os
import datetime
import json
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from meta_utils import get_env_info, get_git_info, save_json


def safe_len(x):
    try:
        return len(x)
    except:
        return -1


def pick_h5(model_dir, model_file=None):
    if model_file:
        return os.path.join(model_dir, model_file)
    h5s = sorted([f for f in os.listdir(model_dir) if f.endswith("_model.h5")])
    if not h5s:
        raise FileNotFoundError(f"No *_model.h5 found in: {model_dir}")
    return os.path.join(model_dir, h5s[0])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--model_file", type=str, default=None)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--test_json", type=str, required=True)
    p.add_argument("--save_root", type=str, default="../results")
    p.add_argument("--repo_root", type=str, default="..")
    p.add_argument("--save_frame_csv", action="store_true",
                   help="save per-frame dataframe with pred_fx/pred_fy (can be big)")
    args = p.parse_args()

    # load train meta -> SEQ_LEN/STRIDE 자동 재현 (사용자 실수 차단)
    meta_path = os.path.join(args.model_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in model_dir: {args.model_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        train_meta = json.load(f)

    SEQ_LEN = int(train_meta["hparams"]["SEQ_LEN"])
    STRIDE = int(train_meta["hparams"]["STRIDE"])
    input_columns = train_meta["data"]["input_columns"]
    output_columns = train_meta["data"]["output_columns"]
    col_size = int(train_meta["data"]["joint_angle_len_expected"])

    time_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_h5 = pick_h5(args.model_dir, args.model_file)

    model = tf.keras.models.load_model(model_h5)
    scaler_x = joblib.load(os.path.join(args.model_dir, "scaler_x.pkl"))
    scaler_y = joblib.load(os.path.join(args.model_dir, "scaler_y.pkl"))

    test_csv = sorted(glob(args.test_csv))
    test_json = sorted(glob(args.test_json))
    if len(test_csv) == 0:
        raise FileNotFoundError(f"No CSV matched: {args.test_csv}")
    if len(test_json) == 0:
        raise FileNotFoundError(f"No JSON matched: {args.test_json}")

    save_dir = os.path.join(args.save_root, f"results_{model.name}", time_now)
    os.makedirs(save_dir, exist_ok=True)

    pred_meta = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "save_dir": os.path.abspath(save_dir),
        "model_dir": os.path.abspath(args.model_dir),
        "model_h5": os.path.abspath(model_h5),
        "model_input_shape": list(model.input_shape) if isinstance(model.input_shape, tuple) else str(model.input_shape),
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "seq_len_source": "meta.json",
        "test": {
            "test_csv_glob": args.test_csv,
            "test_json_glob": args.test_json,
            "num_csv": len(test_csv),
            "num_json": len(test_json),
            "csv_samples": test_csv[:50],
            "json_samples": test_json[:50],
            "pairing_by_zip": True,
        },
        "train_meta_ref": {
            "train_timestamp": train_meta.get("timestamp", None),
            "train_model_type": train_meta.get("model_type", None),
            "train_save_dir": train_meta.get("save_dir", None),
        },
        "env": get_env_info(),
        "git": get_git_info(repo_root=args.repo_root),
    }
    save_json(os.path.join(save_dir, "predict_meta.json"), pred_meta)

    all_pred_rows = []
    all_frame_rows = []
    mismatch_logs = []

    for csv_path, json_path in zip(test_csv, test_json):
        df_csv = pd.read_csv(csv_path).reset_index(drop=True)
        df_json = pd.read_json(json_path).reset_index(drop=True)

        n = min(len(df_csv), len(df_json))
        if len(df_csv) != len(df_json):
            mismatch_logs.append(
                f"{os.path.basename(csv_path)}({len(df_csv)}) vs {os.path.basename(json_path)}({len(df_json)}) -> trunc {n}"
            )
        df_csv = df_csv.iloc[:n].reset_index(drop=True)
        df_json = df_json.iloc[:n].reset_index(drop=True)
        df = pd.concat([df_csv, df_json], axis=1)

        # 깨진 joint angle 제거
        ja_len = df["Joint Angle"].apply(safe_len)
        df = df[ja_len == col_size].reset_index(drop=True)

        joint_angle = np.array(df["Joint Angle"].tolist(), dtype=np.float32)
        ja_df = pd.DataFrame(joint_angle, columns=[f"Joint Angle_{i}" for i in range(col_size)])
        df = pd.concat([df.reset_index(drop=True), ja_df], axis=1)

        df = df.dropna(subset=input_columns + output_columns).reset_index(drop=True)
        T = len(df)
        if T < SEQ_LEN:
            continue

        x_raw = df[input_columns].values.astype(np.float32)   # (T, F)
        y_raw = df[output_columns].values.astype(np.float32)  # (T, 2)

        X_list = []
        end_indices = []
        for s in range(0, T - SEQ_LEN + 1, STRIDE):
            e = s + SEQ_LEN
            X_list.append(x_raw[s:e])
            end_indices.append(e - 1)

        X = np.array(X_list, dtype=np.float32)  # (N, SEQ_LEN, F)
        Xn = scaler_x.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        pred_n = model.predict(Xn, verbose=0)
        pred = scaler_y.inverse_transform(pred_n)

        if args.save_frame_csv:
            df_out = df.copy()
            df_out["pred_fx"] = np.nan
            df_out["pred_fy"] = np.nan

        for k, idx in enumerate(end_indices):
            all_pred_rows.append({
                "src_csv": os.path.basename(csv_path),
                "frame_idx": int(idx),
                "gt_fx": float(y_raw[idx, 0]),
                "gt_fy": float(y_raw[idx, 1]),
                "pred_fx": float(pred[k, 0]),
                "pred_fy": float(pred[k, 1]),
            })
            if args.save_frame_csv:
                df_out.at[idx, "pred_fx"] = float(pred[k, 0])
                df_out.at[idx, "pred_fy"] = float(pred[k, 1])

        if args.save_frame_csv:
            df_out["src_csv"] = os.path.basename(csv_path)
            all_frame_rows.append(df_out)

    pred_df = pd.DataFrame(all_pred_rows)
    pred_save = os.path.join(save_dir, f"predicted_timeseries_results_{time_now}.csv")
    pred_df.to_csv(pred_save, index=False)

    if args.save_frame_csv and len(all_frame_rows):
        frame_df = pd.concat(all_frame_rows, ignore_index=True)
        frame_save = os.path.join(save_dir, f"dataframe_with_pred_all_{time_now}.csv")
        frame_df.to_csv(frame_save, index=False)

    pred_meta["test"]["length_mismatch_logs"] = mismatch_logs[:200]
    save_json(os.path.join(save_dir, "predict_meta.json"), pred_meta)

    print("Saved:", pred_save)


if __name__ == "__main__":
    main()
