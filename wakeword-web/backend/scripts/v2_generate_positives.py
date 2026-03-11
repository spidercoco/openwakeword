import torch
import os
import sys
import uuid
import numpy as np
import argparse
import logging
import soundfile as sf
import random
import yaml
import json
from tqdm import tqdm
from pathlib import Path

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("Error: Qwen3TTSModel not found.")
    exit(1)

# ================= 配置 =================
MODEL_PATH = "/data/model/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# 加载声音描述库
VOICE_DATA = []
try:
    # voices.json 现在就在 scripts 目录下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    voices_path = os.path.join(script_dir, "voices.json")
    if os.path.exists(voices_path):
        with open(voices_path, "r", encoding="utf-8") as f:
            VOICE_DATA = json.load(f)
        print(f"Loaded {len(VOICE_DATA)} voice prompts from {voices_path}")
except Exception as e:
    print(f"Error loading voices.json: {e}")

EMOTIONS = ["平静", "欢快", "温柔", "严肃", "兴奋", "充满活力", "坚定", "亲切"]

def get_random_instruct():
    if VOICE_DATA:
        return random.choice(VOICE_DATA).get("prompt", "自然")
    return random.choice(EMOTIONS)

# ================= 工具函数 =================
def extract_tail(audio, sr, max_duration=1.3):
    max_samples = int(max_duration * sr)
    return audio[-max_samples:] if len(audio) > max_samples else audio

def trim_end_silence(audio, threshold=0.005):
    mask = np.abs(audio) > threshold
    if not np.any(mask): return audio
    return audio[:len(audio) - np.argmax(mask[::-1])]

def generate_samples(text, max_samples, output_dir, batch_size=1, model=None):
    os.makedirs(output_dir, exist_ok=True)
    current_count = 0
    pbar = tqdm(total=max_samples, desc=f"Generating to {os.path.basename(output_dir)}")
    
    while current_count < max_samples:
        current_batch_size = min(batch_size, max_samples - current_count)
        batch_texts = [text] * current_batch_size
        batch_instructs = [get_random_instruct() for _ in range(current_batch_size)]
        
        try:
            # 使用 VoiceDesign 模式生成
            wavs, sr = model.generate_voice_design(
                text=batch_texts, 
                language=["Chinese"] * current_batch_size,
                instruct=batch_instructs
            )
            for i in range(len(wavs)):
                audio_data = wavs[i].cpu().numpy() if torch.is_tensor(wavs[i]) else wavs[i]
                if len(audio_data) / sr > 6.0: continue
                audio_final = extract_tail(trim_end_silence(audio_data), sr=sr, max_duration=1.3)
                if len(audio_final) / sr < 0.3: continue
                
                fname = f"pos_{uuid.uuid4().hex[:10]}.wav"
                sf.write(os.path.join(output_dir, fname), audio_final, sr)
                current_count += 1
                pbar.update(1)
                # 为平台进度显示
                print(f"PROGRESS:{current_count}/{max_samples}", flush=True)
                if current_count >= max_samples: break
        except Exception as e:
            print(f"Error: {e}")
            continue
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # 脚本运行在 task_dir，所有路径基于此
    task_root = os.path.dirname(os.path.abspath(args.config))
    
    # 路径加上 _tts 后缀
    train_dir = os.path.join(task_root, "positive_train_tts")
    test_dir = os.path.join(task_root, "positive_test_tts")

    # 加载模型
    dtype = torch.bfloat16 if "cuda" in DEVICE else (torch.float16 if "mps" in DEVICE else torch.float32)
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=dtype, trust_remote_code=True)

    print(f"Generating training positives...")
    generate_samples(config["target_phrase"], config["n_samples"], train_dir, config.get("tts_batch_size", 1), model)
    
    print(f"Generating testing positives...")
    generate_samples(config["target_phrase"], config["n_samples_val"], test_dir, config.get("tts_batch_size", 1), model)
    
    print("Done.")
