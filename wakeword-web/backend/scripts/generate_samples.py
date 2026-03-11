import os
import uuid
import torch
import random
import argparse
import soundfile as sf
import numpy as np
import json
from pathlib import Path

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("Error: Qwen3TTSModel not found. Ensure the Qwen3-TTS repo is in your PYTHONPATH.")
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

def get_random_instruct():
    if VOICE_DATA:
        return random.choice(VOICE_DATA).get("prompt", "自然")
    return "自然"

def main():
    parser = argparse.ArgumentParser(description="Generate wake word samples with random voices from json")
    parser.add_argument("--wakeword", type=str, required=True, help="The exact wake word to generate")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated samples")
    parser.add_argument("--num_samples", type=int, default=3, help="Total samples to generate")
    parser.add_argument("--speaker", type=str, default=None, help="Ignored")
    
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 加载模型
    dtype = torch.bfloat16 if "cuda" in DEVICE else (torch.float16 if "mps" in DEVICE else torch.float32)
    print(f"Loading Qwen3-TTS on {DEVICE}...")
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=dtype, trust_remote_code=True)

    print(f"Generating samples using random prompts from voices.json for: {args.wakeword}")

    count = 0
    while count < args.num_samples:
        try:
            # 随机从 JSON 中获取音色指令
            instruct = get_random_instruct()

            # 使用 VoiceDesign 模式生成
            wavs, sr = model.generate_voice_design(
                text=[args.wakeword],
                language=["Chinese"],
                instruct=[instruct]
            )

            audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]
            
            # 文件名逻辑
            clean_text = args.wakeword.replace("，", "").replace(" ", "").replace("?", "")
            file_id = f"{clean_text}_{uuid.uuid4().hex[:6]}"
            file_name = f"{file_id}.wav"
            
            sf.write(out_path / file_name, audio_data, sr)
            count += 1
            
            print(f"PROGRESS:{count}/{args.num_samples}")
            print(f"[{count}/{args.num_samples}] Saved: {file_name} with prompt: {instruct[:30]}...", flush=True)

        except Exception as e:
            print(f"Generation error: {e}")
            continue

    print(f"Success. Generated {count} samples in {args.output_dir}")

if __name__ == "__main__":
    main()
