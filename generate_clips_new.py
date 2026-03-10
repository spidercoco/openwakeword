import torch
from torch import optim, nn
import torchinfo
import copy
import os
import sys
import tempfile
import uuid
import numpy as np
import scipy
import collections
import argparse
import logging
import soundfile as sf
import random
from tqdm import tqdm
import yaml
from pathlib import Path
import openwakeword
from openwakeword.data import augment_clips, mmap_batch_generator
from openwakeword.utils import compute_features_from_generator
from openwakeword.utils import AudioFeatures

# Mock generate_adversarial_texts
def generate_adversarial_texts(input_text, N, **kwargs):
    return [input_text] * N

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("Error: Qwen3TTSModel not found. Ensure the Qwen3-TTS repo is in your PYTHONPATH.")
    exit(1)

# ================= Qwen3 TTS 配置 =================
MODEL_PATH = "/data/model/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
PRESET_SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
EMOTIONS = ["平静", "欢快", "温柔", "严肃", "兴奋", "充满活力", "坚定", "亲切"]

# 全局变量存储模型，避免重复加载
QWEN_MODEL = None

def get_qwen_model():
    global QWEN_MODEL
    if QWEN_MODEL is None:
        dtype = torch.bfloat16 if "cuda" in DEVICE else (torch.float16 if "mps" in DEVICE else torch.float32)
        QWEN_MODEL = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=dtype, trust_remote_code=True)
    return QWEN_MODEL

def extract_tail(audio, sr, max_duration=1.3):
    max_samples = int(max_duration * sr)
    if len(audio) > max_samples:
        return audio[-max_samples:]
    return audio

def trim_end_silence(audio, threshold=0.005):
    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio
    end_idx = len(audio) - np.argmax(mask[::-1])
    return audio[:end_idx]

def generate_optimized_instruct():
    emotion = random.choice(EMOTIONS)
    return f"{emotion}地说话".strip()

def generate_samples(text, max_samples, output_dir, batch_size=1, file_names=None, **kwargs):
    """
    使用 Qwen3-TTS 替换原有的 Piper 生成逻辑
    保留 batch_size 参数用于批量生成
    """
    model = get_qwen_model()
    os.makedirs(output_dir, exist_ok=True)
    
    texts = [text] if isinstance(text, str) else text
    
    current_count = 0
    pbar = tqdm(total=max_samples, desc=f"Generating to {os.path.basename(output_dir)}")
    
    while current_count < max_samples:
        # 计算当前 batch 的大小
        current_batch_size = min(batch_size, max_samples - current_count)
        
        # 为当前 batch 准备文本、音色和指令
        batch_texts = [random.choice(texts) for _ in range(current_batch_size)]
        batch_speakers = [random.choice(PRESET_SPEAKERS) for _ in range(current_batch_size)]
        batch_instructs = [generate_optimized_instruct() for _ in range(current_batch_size)]
        
        try:
            # 批量合成
            wavs, sr = model.generate_custom_voice(
                text=batch_texts,
                language=["Chinese"] * current_batch_size,
                speaker=batch_speakers,
                instruct=batch_instructs
            )

            for i in range(len(wavs)):
                audio_data = wavs[i].cpu().numpy() if torch.is_tensor(wavs[i]) else wavs[i]
                
                # 过滤与处理
                if len(audio_data) / sr > 6.0:
                    continue

                audio_no_silence = trim_end_silence(audio_data, threshold=0.005)
                audio_final = extract_tail(audio_no_silence, sr=sr, max_duration=1.3)
                
                if len(audio_final) / sr < 0.3:
                    continue

                if file_names and current_count < len(file_names):
                    fname = file_names[current_count]
                else:
                    fname = f"{batch_speakers[i]}_{uuid.uuid4().hex[:10]}.wav"
                
                sf.write(os.path.join(output_dir, fname), audio_final, sr)
                
                current_count += 1
                pbar.update(1)
                
                # 为以后平台集成提供的实时进度接口
                print(f"PROGRESS:{current_count}/{max_samples}", flush=True)
                
                if current_count >= max_samples:
                    break
            
        except Exception as e:
            logging.error(f"Generation error: {e}")
            continue
    pbar.close()

# ================= 原有逻辑适配 =================

config = yaml.load(open("my_model.yaml", 'r').read(), yaml.Loader)

# Define output locations
config["output_dir"] = os.path.abspath(config["output_dir"])
if not os.path.exists(config["output_dir"]):
    os.mkdir(config["output_dir"])
if not os.path.exists(os.path.join(config["output_dir"], config["model_name"])):
    os.mkdir(os.path.join(config["output_dir"], config["model_name"]))

positive_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "positive_train")
positive_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "positive_test")
negative_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_train")
negative_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_test")
feature_save_dir = os.path.join(config["output_dir"], config["model_name"])

# Get paths for impulse response and background audio files
rir_paths = [i.path for j in config["rir_paths"] for i in os.scandir(j)]
background_paths = []
if len(config["background_paths_duplication_rate"]) != len(config["background_paths"]):
    config["background_paths_duplication_rate"] = [1]*len(config["background_paths"])
for background_path, duplication_rate in zip(config["background_paths"], config["background_paths_duplication_rate"]):
    background_paths.extend([i.path for i in os.scandir(background_path)]*duplication_rate)

# Generate positive clips for training
logging.info("#"*50 + "\nGenerating positive clips for training\n" + "#"*50)
if not os.path.exists(positive_train_output_dir):
    os.mkdir(positive_train_output_dir)
n_current_samples = len(os.listdir(positive_train_output_dir))
if n_current_samples <= 0.95*config["n_samples"]:
    generate_samples(
        text=config["target_phrase"], max_samples=config["n_samples"]-n_current_samples,
        batch_size=config["tts_batch_size"],
        output_dir=positive_train_output_dir,
        file_names=[uuid.uuid4().hex + ".wav" for i in range(config["n_samples"])]
    )
    if DEVICE == "cuda": torch.cuda.empty_cache()
else:
    logging.warning(f"Skipping generation of positive clips for training, as ~{config['n_samples']} already exist")

# Generate positive clips for testing
logging.info("#"*50 + "\nGenerating positive clips for testing\n" + "#"*50)
if not os.path.exists(positive_test_output_dir):
    os.mkdir(positive_test_output_dir)
n_current_samples = len(os.listdir(positive_test_output_dir))
if n_current_samples <= 0.95*config["n_samples_val"]:
    generate_samples(text=config["target_phrase"], max_samples=config["n_samples_val"]-n_current_samples,
                     batch_size=config["tts_batch_size"],
                     output_dir=positive_test_output_dir)
    if DEVICE == "cuda": torch.cuda.empty_cache()
else:
    logging.warning(f"Skipping generation of positive clips testing, as ~{config['n_samples_val']} already exist")

# Generate adversarial negative clips for training
logging.info("#"*50 + "\nGenerating negative clips for training\n" + "#"*50)
if not os.path.exists(negative_train_output_dir):
    os.mkdir(negative_train_output_dir)
n_current_samples = len(os.listdir(negative_train_output_dir))
if n_current_samples <= 0.95*config["n_samples"]:
    adversarial_texts = config["custom_negative_phrases"]
    target_phrases = [config["target_phrase"]] if isinstance(config["target_phrase"], str) else config["target_phrase"]
    for target_phrase in target_phrases:
        adversarial_texts.extend(generate_adversarial_texts(
            input_text=target_phrase,
            N=config["n_samples"]//len(target_phrases)))
    generate_samples(text=adversarial_texts, max_samples=config["n_samples"]-n_current_samples,

                     batch_size=config["tts_batch_size"],
                     output_dir=negative_train_output_dir,
                     file_names=[uuid.uuid4().hex + ".wav" for i in range(config["n_samples"])]
                     )
    if DEVICE == "cuda": torch.cuda.empty_cache()
else:
    logging.warning(f"Skipping generation of negative clips for training, as ~{config['n_samples']} already exist")

# Generate adversarial negative clips for testing
logging.info("#"*50 + "\nGenerating negative clips for testing\n" + "#"*50)
if not os.path.exists(negative_test_output_dir):
    os.mkdir(negative_test_output_dir)
n_current_samples = len(os.listdir(negative_test_output_dir))
if n_current_samples <= 0.95*config["n_samples_val"]:
    adversarial_texts = config["custom_negative_phrases"]
    target_phrases = [config["target_phrase"]] if isinstance(config["target_phrase"], str) else config["target_phrase"]
    for target_phrase in target_phrases:
        adversarial_texts.extend(generate_adversarial_texts(
            input_text=target_phrase,
            N=config["n_samples_val"]//len(target_phrases)))
    generate_samples(text=adversarial_texts, max_samples=config["n_samples_val"]-n_current_samples,

                     batch_size=config["tts_batch_size"],
                     output_dir=negative_test_output_dir)
    if DEVICE == "cuda": torch.cuda.empty_cache()
else:
    logging.warning(f"Skipping generation of negative clips for testing, as ~{config['n_samples_val']} already exist")
