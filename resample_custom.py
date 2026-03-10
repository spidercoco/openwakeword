import os
import yaml
import subprocess
import shutil
from tqdm import tqdm
from pathlib import Path

def resample_directory(input_dir, output_dir, target_sr=16000):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob("*.wav"))
    if not files:
        print(f"No .wav files found in {input_dir}")
        return False

    print(f"Resampling {len(files)} files to {target_sr}Hz...")

    for f in tqdm(files):
        output_file = output_path / f.name
        cmd = [
            "ffmpeg", "-y", "-i", str(f),
            "-ar", str(target_sr), "-ac", "1",
            str(output_file)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            return False
    return True

if __name__ == "__main__":
    config_path = "my_model.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        exit(1)
        
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    base_output = os.path.expanduser(config["output_dir"])
    model_name = config["model_name"]
    target_root = os.path.join(base_output, model_name)

    target_dirs = ["positive_train", "positive_test", "negative_train", "negative_test"]

    print(f"Target root: {os.path.abspath(target_root)}")

    for dname in target_dirs:
        orig_path = os.path.join(target_root, dname)
        tmp_16k_path = os.path.join(target_root, f"{dname}_tmp_16k")
        bak_path = os.path.join(target_root, f"{dname}_bak")

        if os.path.exists(orig_path):
            print(f"\n--- Processing: {dname} ---")
            # 1. 执行重采样到临时目录
            success = resample_directory(orig_path, tmp_16k_path)
            
            if success:
                # 2. 如果备份已存在，先删除旧备份
                if os.path.exists(bak_path):
                    print(f"Removing old backup: {bak_path}")
                    shutil.rmtree(bak_path)
                
                # 3. 交换目录
                print(f"Renaming {dname} -> {dname}_bak")
                os.rename(orig_path, bak_path)
                
                print(f"Renaming {dname}_tmp_16k -> {dname}")
                os.rename(tmp_16k_path, orig_path)
                
                print(f"Success! Original files backed up in {dname}_bak")
            else:
                print(f"Error occurred during resampling {dname}. No changes made.")
                if os.path.exists(tmp_16k_path):
                    shutil.rmtree(tmp_16k_path)
        else:
            print(f"\nSkipping {dname}: path does not exist")

    print("\nAll tasks complete.")
