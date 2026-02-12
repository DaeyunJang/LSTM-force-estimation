from pathlib import Path
import os
import numpy as np
import tensorflow as tf


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    x = float(n)
    for u in units:
        if x < 1024:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} TB"


def count_params(model):
    total = model.count_params()
    trainable = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    non_trainable = total - trainable
    return total, trainable, non_trainable


def find_all_h5(root_dir: str):
    # **여기가 핵심**: rglob → 모든 하위 폴더 탐색
    return sorted(Path(root_dir).rglob("*.h5"))


def summarize_h5_models(root_dir: str):
    h5_paths = find_all_h5(root_dir)

    print(f"Found {len(h5_paths)} h5 files\n")
    print(f"{'model':25s}  {'file':25}  {'#Params':>12s}  {'ParamMem(fp32)':>14s}  {'FileSize':>10s}")
    print("-" * 115)

    for p in h5_paths:
        try:
            model = tf.keras.models.load_model(p, compile=False)
            total, _, _ = count_params(model)
            param_mem = total * 4

            print(f"{p.parent.name:25s}  {p.name:40s}  {total:12d}  {human_bytes(param_mem):>14s}  {human_bytes(os.path.getsize(p)):>10s}")

        except Exception as e:
            print(f"{p.parent.name:25s}  {p.name:40s}  {'LOAD FAIL':>12s}  {'-':>14s}  {human_bytes(os.path.getsize(p)):>10s}")
            print(f"  -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    summarize_h5_models("../fit_2k_angle")
