import os
import uuid
import torch
import random
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("Error: Qwen3TTSModel not found. Ensure the Qwen3-TTS repo is in your PYTHONPATH.")
    exit(1)

# ================= 配置 =================
CORE_WAKE_WORD = "小熊"
# 强制加上前缀，确保“小熊”在音频末尾，方便截断前面的幻觉/前置词
WAKE_WORD_VARIATIONS = [
    "你好小熊",
    "嘿小熊",
    "喂小熊",
    "唤醒小熊",
    "请帮我叫小熊",
    "好的小熊",
    "我在小熊",
    "收到小熊"
]

MODEL_PATH = "/Users/jingyehuang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_DIR = "wake_word_dataset_5000"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_SAMPLES_TARGET = 5000

# 预设音色
PRESET_SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
EMOTIONS = ["平静", "欢快", "温柔", "严肃", "兴奋", "充满活力", "坚定", "亲切"]

# ================= 核心处理函数 =================

def extract_tail(audio, sr, max_duration=1.3):
    """
    核心策略：既然“小熊”在句末，截取最后 1.3s 足以包含完整的发音，
    同时能有效剔除模型在前半部分产生的幻觉或前置词。
    """
    max_samples = int(max_duration * sr)
    if len(audio) > max_samples:
        return audio[-max_samples:]
    return audio

def trim_end_silence(audio, threshold=0.005):
    """修剪末尾的纯静音点"""
    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio
    end_idx = len(audio) - np.argmax(mask[::-1])
    return audio[:end_idx]

def generate_optimized_instruct():
    emotion = random.choice(EMOTIONS)
    return f"{emotion}地说话".strip()

# ================= 主逻辑 =================

def main():
    out_path = Path(OUTPUT_DIR)
    wavs_path = out_path / "wavs"
    wavs_path.mkdir(parents=True, exist_ok=True)
    metadata_csv = out_path / "metadata.csv"

    existing_files = set()
    if metadata_csv.exists():
        with open(metadata_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split("|")
                if parts: existing_files.add(parts[0])

    current_count = len(existing_files)
    print(f"Current valid samples: {current_count}")
    
    if current_count >= TOTAL_SAMPLES_TARGET:
        print(f"Dataset already complete.")
        return

    # 加载模型
    dtype = torch.bfloat16 if "cuda" in DEVICE else (torch.float16 if "mps" in DEVICE else torch.float32)
    print(f"Loading Qwen3-TTS on {DEVICE}...")
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=dtype, trust_remote_code=True)

    if not metadata_csv.exists():
        with open(metadata_csv, "w", encoding="utf-8") as f:
            f.write("file_name|text|speaker|instruct|original_duration|final_duration\n")

    with tqdm(total=TOTAL_SAMPLES_TARGET, initial=current_count, desc="Generating") as pbar:
        while current_count < TOTAL_SAMPLES_TARGET:
            spk = random.choice(PRESET_SPEAKERS)
            text = random.choice(WAKE_WORD_VARIATIONS)
            instruct = generate_optimized_instruct()
            
            try:
                # 合成
                wav, sr = model.generate_custom_voice(
                    text=text,
                    language="Chinese",
                    speaker=spk,
                    instruct=instruct
                )

                audio_data = wav[0].cpu().numpy() if torch.is_tensor(wav) else wav[0]
                orig_duration = len(audio_data) / sr
                
                # 如果总长太夸张 (> 6s)，说明幻觉严重，舍弃
                if orig_duration > 6.0:
                    continue

                # 1. 修剪末尾静音
                audio_no_silence = trim_end_silence(audio_data, threshold=0.005)
                
                # 2. 截取最后 1.3 秒
                audio_final = extract_tail(audio_no_silence, sr=sr, max_duration=1.3)
                
                final_duration = len(audio_final) / sr
                
                if final_duration < 0.3:
                    continue

                file_id = f"{spk}_{uuid.uuid4().hex[:10]}"
                file_name = f"{file_id}.wav"
                
                sf.write(wavs_path / file_name, audio_final, sr)
                
                # 写入 Metadata
                with open(metadata_csv, "a", encoding="utf-8") as f:
                    f.write(f"{file_name}|{text}|{spk}|{instruct}|{orig_duration:.2f}|{final_duration:.2f}\n")
                
                current_count += 1
                pbar.update(1)
                
            except Exception:
                continue

    print(f"\nSuccess! Generated {current_count} samples in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
