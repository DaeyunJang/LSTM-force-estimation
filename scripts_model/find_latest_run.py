import sys
from pathlib import Path

def latest_subdir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    subs = [p for p in path.iterdir() if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No run folders in: {path}")
    return sorted(subs, key=lambda p: p.name)[-1]

def main():
    if len(sys.argv) != 2:
        print("Usage: python find_latest_run.py <fit_model_root_dir>")
        sys.exit(1)
    root = Path(sys.argv[1])
    print(str(latest_subdir(root)))

if __name__ == "__main__":
    main()
