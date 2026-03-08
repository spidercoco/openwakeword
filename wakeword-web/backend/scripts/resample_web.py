import os
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

def resample_audio(input_dir, output_dir, target_sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    wav_files = list(Path(input_dir).glob("*.wav"))
    total = len(wav_files)
    
    print(f"Resampling {total} files from {input_dir} to {output_dir}...")
    
    for i, wav_path in enumerate(wav_files):
        output_path = Path(output_dir) / wav_path.name
        # 参考 resample.py 的 ffmpeg 命令
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-ar", str(target_sr), "-ac", "1",
            str(output_path)
        ]
        # 执行转换
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 为 Web 打印进度
        if (i + 1) % 5 == 0 or (i + 1) == total:
            print(f"PROGRESS:{i+1}/{total}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    resample_audio(args.input_dir, args.output_dir)
