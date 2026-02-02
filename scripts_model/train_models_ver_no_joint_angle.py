import os
import datetime
from glob import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from model_zoo import build_model, get_model_spec
from meta_utils import get_env_info, get_git_info, save_json, save_model_summary, save_history


def safe_len(x):
    try:
        return len(x)
    except:
        return -1



def build_sequences_from_files(csv_files, json_files, input_cols, output_cols, seq_len, stride):
    X_list, y_list = [], []
    mismatch_logs = []

    for csv_path, json_path in zip(csv_files, json_files):
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

        # NaN 제거
        df = df.dropna(subset=input_cols + output_cols).reset_index(drop=True)

        x_raw = df[input_cols].values.astype(np.float32)  # (T, F)
        y_raw = df[output_cols].values.astype(np.float32) # (T, 2)

        T = len(df)
        if T < seq_len:
            continue

        for s in range(0, T - seq_len + 1, stride):
            e = s + seq_len
            X_list.append(x_raw[s:e])
            y_list.append(y_raw[e-1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, mismatch_logs


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="mlp|cnn|tcn|gru|lstm|transformer|kalmannet|convmixer|resnet")
    p.add_argument("--train_csv", type=str, default="../datasets/train/data_*.csv")
    p.add_argument("--train_json", type=str, default="../datasets/train/curve_fit_result-joint_angle_*.json")
    p.add_argument("--save_root", type=str, default="../fit")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--repo_root", type=str, default="..")
    args = p.parse_args()

    model_name = args.model.lower()
    spec = get_model_spec(model_name)
    SEQ_LEN = int(spec["seq_len"])
    STRIDE = int(spec["stride"])

    # save dir
    time_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_root, f"fit_{model_name.upper()}", time_now)
    os.makedirs(save_dir, exist_ok=True)

    # file list
    csv_files = sorted(glob(args.train_csv))
    json_files = sorted(glob(args.train_json))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV matched: {args.train_csv}")
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON matched: {args.train_json}")

    # input/output columns (Joint Angle 미사용 버전)
    input_cols = ["wire length #0", "wire length #1", "loadcell #0", "loadcell #1"]
    output_cols = ["fx", "fy"]
    # output_cols = ["fx_kalman", "fy_kalman"]

    # build sequences
    X, y, mismatch_logs = build_sequences_from_files(
        csv_files, json_files,
        input_cols=input_cols,
        output_cols=output_cols,
        seq_len=SEQ_LEN,
        stride=STRIDE
    )
    if X.shape[0] == 0:
        raise RuntimeError(f"No sequences built. Check data or reduce SEQ_LEN (currently {SEQ_LEN}).")

    # split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # normalize (train 기준)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_valid_flat = X_valid.reshape(-1, X_valid.shape[-1])

    scaler_x.fit(X_train_flat)
    X_train_n = scaler_x.transform(X_train_flat).reshape(X_train.shape)
    X_valid_n = scaler_x.transform(X_valid_flat).reshape(X_valid.shape)

    scaler_y.fit(y_train)
    y_train_n = scaler_y.transform(y_train)
    y_valid_n = scaler_y.transform(y_valid)

    # joblib.dump(scaler_x, os.path.join(save_dir, "scaler_x.pkl"))
    # joblib.dump(scaler_y, os.path.join(save_dir, "scaler_y.pkl"))
    with open(os.path.join(save_dir, "scaler_x.pkl"), "wb") as f:
        pickle.dump(scaler_x, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_dir, "scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f, protocol=pickle.HIGHEST_PROTOCOL)

    # model
    input_shape = (SEQ_LEN, X_train_n.shape[-1])
    model = build_model(model_name, input_shape=input_shape)

    if args.loss == "huber":
        loss_fn = tf.keras.losses.Huber(delta=args.huber_delta)
    else:
        loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=Adam(learning_rate=args.lr), loss=loss_fn, metrics=["mae"])

    # save summary
    save_model_summary(model, os.path.join(save_dir, "model_summary.txt"))

    # meta.json (train)
    meta = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "save_dir": os.path.abspath(save_dir),
        "model_type": model_name,
        "hparams": {
            "SEQ_LEN": SEQ_LEN,
            "STRIDE": STRIDE,
            "SEQ_LEN_SOURCE": "MODEL_SPECS",
            "EPOCHS": args.epochs,
            "BATCH_SIZE": args.batch,
            "LEARNING_RATE": args.lr,
            "LOSS_NAME": args.loss,
            "HUBER_DELTA": args.huber_delta,
        },
        "data": {
            "train_csv_glob": args.train_csv,
            "train_json_glob": args.train_json,
            "num_csv": len(csv_files),
            "num_json": len(json_files),
            "csv_samples": csv_files[:20],
            "json_samples": json_files[:20],
            "pairing_by_zip": True,
            "length_mismatch_logs": mismatch_logs[:200],
            "input_columns": input_cols,
            "output_columns": output_cols,
            },
        "shapes": {
            "X": list(X.shape),
            "y": list(y.shape),
            "X_train": list(X_train.shape),
            "X_valid": list(X_valid.shape),
            "y_train": list(y_train.shape),
            "y_valid": list(y_valid.shape),
        },
        "env": get_env_info(),
        "git": get_git_info(repo_root=args.repo_root),
        "model": {
            "keras_input_shape": list(model.input_shape) if isinstance(model.input_shape, tuple) else str(model.input_shape),
        }
    }
    save_json(os.path.join(save_dir, "meta.json"), meta)

    # callbacks
    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1
    )
    tb = tf.keras.callbacks.TensorBoard(log_dir=save_dir, histogram_freq=1)

    # train
    history = model.fit(
        X_train_n, y_train_n,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_data=(X_valid_n, y_valid_n),
        callbacks=[tb, es, reduce_lr],
        verbose=1
    )
    save_history(history, os.path.join(save_dir, "history.json"))

    # eval
    val_loss, val_mae = model.evaluate(X_valid_n, y_valid_n, verbose=1)

    # update meta
    meta["result"] = {
        "final_val_loss": float(val_loss),
        "final_val_mae": float(val_mae),
        "best_val_loss": float(np.min(history.history.get("val_loss", [np.nan]))),
        "best_epoch_by_val_loss": int(np.argmin(history.history.get("val_loss", [np.inf]))) if "val_loss" in history.history else None,
    }
    save_json(os.path.join(save_dir, "meta.json"), meta)

    # save model h5
    h5_name = f"{model_name}_model.h5"
    model.save(os.path.join(save_dir, h5_name))

    # save valid pred/gt
    pred_n = model.predict(X_valid_n, verbose=0)
    pred = scaler_y.inverse_transform(pred_n)
    gt = scaler_y.inverse_transform(y_valid_n)

    np.savetxt(os.path.join(save_dir, "predicted_value.csv"), pred, delimiter=",")
    np.savetxt(os.path.join(save_dir, "original_value.csv"), gt, delimiter=",")
    gt_pred = np.hstack([gt, pred])
    np.savetxt(
        os.path.join(save_dir, "gt_and_pred.csv"),
        gt_pred, delimiter=",",
        header="gt_fx,gt_fy,pred_fx,pred_fy",
        comments=""
    )

    print("Saved:", save_dir)


if __name__ == "__main__":
    main()