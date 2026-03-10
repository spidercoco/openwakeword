import os
import yaml
import subprocess
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def resample_directory(input_dir, output_dir, target_sr=16000):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Skipping: {input_dir} not found.")
        return False

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob("*.wav"))
    if not files:
        print(f"No wav files in {input_dir}")
        return True

    print(f"Resampling {len(files)} files from {input_dir} to {output_dir}...")

    for i, f in enumerate(files):
        output_file = output_path / f.name
        # ffmpeg 转换：采样率 16000，声道 1 (mono)
        cmd = [
            "ffmpeg", "-y", "-i", str(f),
            "-ar", str(target_sr), "-ac", "1",
            str(output_file)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue
        
        # 为平台输出进度
        if (i + 1) % 5 == 0 or (i + 1) == len(files):
            print(f"PROGRESS:{i+1}/{len(files)}", flush=True)
            
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # 1. 加载配置
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    task_root = os.path.dirname(os.path.abspath(args.config))
    
    # 2. 定义处理映射：{输入目录: 输出目录}
    # 输入是带 _tts 的，输出是正式的不带后缀的
    dirs_to_process = [
        ("positive_train", "positive_train_tts"),
        ("positive_test", "positive_test_tts"),
        ("negative_train", "negative_train_tts"),
        ("negative_test", "negative_test_tts")
    ]

    print(f"Starting resampling for task: {task_root}")

    # 3. 循环处理
    for formal_name, tts_name in dirs_to_process:
        input_path = os.path.join(task_root, tts_name)
        output_path = os.path.join(task_root, formal_name)
        
        if os.path.exists(input_path):
            print(f"\n--- Processing {tts_name} -> {formal_name} ---")
            resample_directory(input_path, output_path)
        else:
            print(f"Skipping {tts_name}, path not found.")

    print("\nAll resampling tasks complete.")
