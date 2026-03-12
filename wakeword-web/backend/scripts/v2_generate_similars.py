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
import shutil
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

def generate_samples(similar_words, max_samples, output_dir, batch_size=1, model=None, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    if not similar_words:
        print(f"Warning: No similar words provided for {output_dir}")
        return

    if overwrite:
        print(f"Overwrite mode: Clearing {output_dir}...")
        for f in os.listdir(output_dir):
            if f.endswith(".wav"): os.remove(os.path.join(output_dir, f))

    # 初次检查
    current_total = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    if current_total >= max_samples:
        print(f"Target reached: {current_total}/{max_samples}. Skipping.")
        print(f"PROGRESS:{max_samples}/{max_samples}", flush=True)
        return

    pbar = tqdm(total=max_samples, initial=current_total, desc=f"Generating to {os.path.basename(output_dir)}")
    
    while True:
        # 实时统计物理文件数
        current_total = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
        print(f"PROGRESS:{current_total}/{max_samples}", flush=True)
        pbar.n = min(current_total, max_samples)
        pbar.refresh()

        if current_total >= max_samples:
            break

        current_batch_size = min(batch_size, max_samples - current_total)
        batch_texts = [random.choice(similar_words) for _ in range(current_batch_size)]
        batch_instructs = [get_random_instruct() for _ in range(current_batch_size)]
        
        try:
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
                
                fname = f"sim_{uuid.uuid4().hex[:10]}.wav"
                sf.write(os.path.join(output_dir, fname), audio_final, sr)
        except Exception as e:
            print(f"Error: {e}")
            continue
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Clear output directory before starting")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    task_root = os.path.dirname(os.path.abspath(args.config))
    train_dir = os.path.join(task_root, "negative_train_tts")
    test_dir = os.path.join(task_root, "negative_test_tts")

    dtype = torch.bfloat16 if "cuda" in DEVICE else (torch.float16 if "mps" in DEVICE else torch.float32)
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=dtype, trust_remote_code=True)

    sim_words = config.get("similar_phrases", [])
    
    print(f"Generating training similars (words: {len(sim_words)})...")
    generate_samples(sim_words, config["n_samples"], train_dir, config.get("tts_batch_size", 1), model, overwrite=args.overwrite)
    
    print(f"Generating testing similars (words: {len(sim_words)})...")
    generate_samples(sim_words, config["n_samples_val"], test_dir, config.get("tts_batch_size", 1), model, overwrite=args.overwrite)
    
    print("Done.")
