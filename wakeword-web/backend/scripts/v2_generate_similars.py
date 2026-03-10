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
from tqdm import tqdm
from pathlib import Path

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("Error: Qwen3TTSModel not found.")
    exit(1)

# ================= 配置 =================
MODEL_PATH = "/data/model/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
PRESET_SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
EMOTIONS = ["平静", "欢快", "温柔", "严肃", "兴奋", "充满活力", "坚定", "亲切"]

# ================= 工具函数 =================
def extract_tail(audio, sr, max_duration=1.3):
    max_samples = int(max_duration * sr)
    return audio[-max_samples:] if len(audio) > max_samples else audio

def trim_end_silence(audio, threshold=0.005):
    mask = np.abs(audio) > threshold
    if not np.any(mask): return audio
    return audio[:len(audio) - np.argmax(mask[::-1])]

def generate_samples(similar_words, max_samples, output_dir, batch_size=1, model=None):
    os.makedirs(output_dir, exist_ok=True)
    if not similar_words:
        print(f"Warning: No similar words provided for {output_dir}")
        return

    current_count = 0
    pbar = tqdm(total=max_samples, desc=f"Generating to {os.path.basename(output_dir)}")
    
    while current_count < max_samples:
        current_batch_size = min(batch_size, max_samples - current_count)
        # 从用户确认的近似词列表中随机抽取
        batch_texts = [random.choice(similar_words) for _ in range(current_batch_size)]
        batch_speakers = [random.choice(PRESET_SPEAKERS) for _ in range(current_batch_size)]
        batch_instructs = [f"{random.choice(EMOTIONS)}地说话" for _ in range(current_batch_size)]
        
        try:
            wavs, sr = model.generate_custom_voice(
                text=batch_texts, language=["Chinese"] * current_batch_size,
                speaker=batch_speakers, instruct=batch_instructs
            )
            for i in range(len(wavs)):
                audio_data = wavs[i].cpu().numpy() if torch.is_tensor(wavs[i]) else wavs[i]
                if len(audio_data) / sr > 6.0: continue
                audio_final = extract_tail(trim_end_silence(audio_data), sr=sr, max_duration=1.3)
                if len(audio_final) / sr < 0.3: continue
                
                fname = f"sim_{uuid.uuid4().hex[:10]}.wav"
                sf.write(os.path.join(output_dir, fname), audio_final, sr)
                current_count += 1
                pbar.update(1)
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

    task_root = os.path.dirname(os.path.abspath(args.config))
    
    # 近似词生成的目录
    train_dir = os.path.join(task_root, "negative_train_tts")
    test_dir = os.path.join(task_root, "negative_test_tts")

    # 加载模型
    dtype = torch.bfloat16 if "cuda" in DEVICE else (torch.float16 if "mps" in DEVICE else torch.float32)
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=dtype, trust_remote_code=True)

    sim_words = config.get("similar_phrases", [])
    
    print(f"Generating training similars (words: {len(sim_words)})...")
    generate_samples(sim_words, config["n_samples"], train_dir, config.get("tts_batch_size", 1), model)
    
    print(f"Generating testing similars (words: {len(sim_words)})...")
    generate_samples(sim_words, config["n_samples_val"], test_dir, config.get("tts_batch_size", 1), model)
    
    print("Done.")
