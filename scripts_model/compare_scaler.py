import joblib
import pandas as pd
import numpy as np

def summarize_scaler(scaler, name):
    summary = {}
    for k in ["mean_", "scale_", "var_", "min_", "data_min_", "data_max_", "data_range_"]:
        if hasattr(scaler, k):
            summary[k] = getattr(scaler, k)
    df = pd.DataFrame(summary)
    print(f"\n===== {name} =====")
    print(df.head())
    print("...")
    print(df.tail())
    print("shape:", df.shape)
    return df

def compare(df_a, df_b, atol=1e-8):
    # 컬럼/shape 다르면 바로 종료
    if list(df_a.columns) != list(df_b.columns) or df_a.shape != df_b.shape:
        print("[DIFF] shape/columns mismatch")
        print("A:", df_a.shape, df_a.columns)
        print("B:", df_b.shape, df_b.columns)
        return

    diff = (df_a - df_b).abs()
    print("\n===== ABS DIFF SUMMARY =====")
    print(diff.describe())
    print("max abs diff:", diff.max().max())
    if np.allclose(df_a.values, df_b.values, atol=atol):
        print("[OK] identical within atol =", atol)
    else:
        print("[DIFF] not identical within atol =", atol)

if __name__ == "__main__":
    p1 = "../fit/fit_LSTM/20260116-082501/scaler_x.pkl"
    p2 = "../fit/fit_MLP/20260116-082435/scaler_x.pkl"

    s1 = joblib.load(p1)
    s2 = joblib.load(p2)

    d1 = summarize_scaler(s1, "LSTM scaler_x")
    d2 = summarize_scaler(s2, "MLP  scaler_x")

    compare(d1, d2)
