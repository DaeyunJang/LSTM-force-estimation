import os
import json
import glob

BASE_DIR = r"..\datasets\train"
json_files = sorted(glob.glob(os.path.join(BASE_DIR, "curve_fit_result-joint_angle_*.json")))

print(f"Total JSON files: {len(json_files)}\n")

for path in json_files:
    print("="*80)
    print("FILE:", os.path.basename(path))

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("JSON LOAD FAILED:", e)
        continue

    bad = []
    for i, row in enumerate(data):
        ja = row.get("Joint Angle", None)
        if not isinstance(ja, list) or len(ja) != 9:
            bad.append((i, ja))

    print(f"Total rows : {len(data)}")
    print(f"Broken rows: {len(bad)}")

    if bad:
        print("First 20 broken indices:")
        for idx, val in bad[:20]:
            print(f"  [{idx}] -> {val}")
    else:
        print("No broken rows found.")
    print()
