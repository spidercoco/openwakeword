import os
import subprocess
from tqdm import tqdm
from pathlib import Path

def resample_directory(input_dir, output_dir, target_sr=16000):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {output_path}")

    files = list(input_path.glob("*.wav"))
    print(f"Found {len(files)} files to resample to {target_sr}Hz (mono)...")

    for f in tqdm(files):
        output_file = output_path / f.name
        # ffmpeg 转换：采样率 16000，声道 1 (mono)
        cmd = [
            "ffmpeg", "-y", "-i", str(f),
            "-ar", str(target_sr),
            "-ac", "1",
            str(output_file)
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    input_dir = "wake_word_dataset_5000/wavs"
    output_dir = "wake_word_dataset_5000/wavs_new"
    
    if os.path.exists(input_dir):
        resample_directory(input_dir, output_dir)
        print(f"Resampling complete! Files are in {output_dir}")
    else:
        print(f"Directory {input_dir} not found.")
