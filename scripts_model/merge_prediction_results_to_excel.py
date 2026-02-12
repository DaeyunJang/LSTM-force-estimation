import re
from pathlib import Path
import pandas as pd


def parse_model_name(dir_name: str) -> str:
    # results_CNN → CNN
    m = re.match(r"results_(.+)", dir_name)
    return m.group(1) if m else dir_name


def pick_latest_run_dir(model_root: Path) -> Path:
    run_dirs = [p for p in model_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run folders under: {model_root}")
    run_dirs.sort(key=lambda p: p.name)
    return run_dirs[-1]


def find_prediction_csv(run_dir: Path) -> Path:
    """
    run_dir 안에서 pred_fx, pred_fy 컬럼을 가진 CSV 자동 탐색
    """
    csv_files = list(run_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in: {run_dir}")

    for csv in csv_files:
        try:
            df = pd.read_csv(csv, nrows=1)
            cols = [c.lower() for c in df.columns]
            if "pred_fx" in cols and "pred_fy" in cols:
                return csv
        except Exception:
            continue

    raise FileNotFoundError(
        f"No CSV with pred_fx/pred_fy columns found in: {run_dir}"
    )


def merge_results(results_root: Path, output_xlsx: Path):
    results_root = Path(results_root)
    model_dirs = [p for p in results_root.iterdir()
                  if p.is_dir() and p.name.startswith("results_")]

    if not model_dirs:
        raise FileNotFoundError(f"No results_* folders under: {results_root}")

    model_dirs.sort(key=lambda p: p.name)

    merged_df = None

    for model_root in model_dirs:
        model_name = parse_model_name(model_root.name)
        run_dir = pick_latest_run_dir(model_root)
        csv_path = find_prediction_csv(run_dir)

        df = pd.read_csv(csv_path)

        # 컬럼 표준화
        cols = {c.lower(): c for c in df.columns}
        gt_fx = cols["gt_fx"]
        gt_fy = cols["gt_fy"]
        pred_fx = cols["pred_fx"]
        pred_fy = cols["pred_fy"]

        if merged_df is None:
            merged_df = df[[gt_fx, gt_fy]].copy()
            merged_df.columns = ["gt_fx", "gt_fy"]

        min_len = min(len(merged_df), len(df))
        merged_df = merged_df.iloc[:min_len].reset_index(drop=True)
        df = df.iloc[:min_len].reset_index(drop=True)

        merged_df[f"pred_fx_{model_name}"] = df[pred_fx]
        merged_df[f"pred_fy_{model_name}"] = df[pred_fy]

        print(f"[OK] {model_name} ← {csv_path.name}")

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_excel(output_xlsx, index=False)

    print(f"\n✅ Saved merged Excel: {output_xlsx.resolve()}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=str, default="../results")
    p.add_argument("--output", type=str, default="../results/merged_results.xlsx")
    args = p.parse_args()

    merge_results(
        results_root=Path(args.results_root),
        output_xlsx=Path(args.output),
    )
