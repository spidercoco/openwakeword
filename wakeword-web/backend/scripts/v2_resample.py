import os
import yaml
import subprocess
import shutil
import argparse
from pathlib import Path

def resample_directory(input_dir, output_dir, target_sr=16000):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob("*.wav"))
    if not files: return True

    for i, f in enumerate(files):
        output_file = output_path / f.name
        cmd = ["ffmpeg", "-y", "-i", str(f), "-ar", str(target_sr), "-ac", "1", str(output_file)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 实时进度
        print(f"PROGRESS:{i+1}/{len(files)}", flush=True)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to task config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    target_root = os.path.dirname(os.path.abspath(args.config))
    target_dirs = ["positive_train", "positive_test", "negative_train", "negative_test"]

    for dname in target_dirs:
        orig_path = os.path.join(target_root, dname)
        if os.path.exists(orig_path):
            tmp_path = os.path.join(target_root, f"{dname}_tmp")
            bak_path = os.path.join(target_root, f"{dname}_bak")
            
            if resample_directory(orig_path, tmp_path):
                if os.path.exists(bak_path): shutil.rmtree(bak_path)
                os.rename(orig_path, bak_path)
                os.rename(tmp_path, orig_path)
