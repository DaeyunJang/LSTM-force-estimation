import os, sys, json, platform, subprocess
import tensorflow as tf
import numpy as np

def _run_cmd(cmd, cwd=None):
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"[ERR] {e}"


def get_git_info(repo_root=".."):
    return {
        "commit": _run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "branch": _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root),
        "status_short": _run_cmd(["git", "status", "--porcelain"], cwd=repo_root),
        "remote": _run_cmd(["git", "remote", "-v"], cwd=repo_root),
    }


def get_env_info():
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "tf_built_with_cuda": bool(tf.test.is_built_with_cuda()),
        "physical_gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "physical_cpus": [d.name for d in tf.config.list_physical_devices("CPU")],
    }
    try:
        env["tf_build_info"] = tf.sysconfig.get_build_info()
    except Exception:
        env["tf_build_info"] = {}
    return env


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_model_summary(model, path_txt):
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_history(history, path_json):
    save_json(path_json, history.history)
