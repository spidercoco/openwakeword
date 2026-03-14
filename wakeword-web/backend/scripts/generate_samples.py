import os
import uuid
import torch
import random
import argparse
import soundfile as sf
import numpy as np
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
        import json
        with open(voices_path, "r", encoding="utf-8") as f:
            VOICE_DATA = json.load(f)
except: pass

def get_random_instruct():
    if VOICE_DATA:
        return random.choice(VOICE_DATA).get("prompt", "自然")
    return "自然"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wakeword", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if "cuda" in DEVICE else (torch.float16 if "mps" in DEVICE else torch.float32)
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=dtype, trust_remote_code=True)

    # 准备批量参数
    batch_texts = [args.wakeword] * args.num_samples
    batch_instructs = [get_random_instruct() for _ in range(args.num_samples)]

    try:
        # 批量生成
        wavs, sr = model.generate_voice_design(
            text=batch_texts,
            language=["Chinese"] * args.num_samples,
            instruct=batch_instructs
        )

        for i, wav in enumerate(wavs):
            audio_data = wav.cpu().numpy() if torch.is_tensor(wav) else wav
            file_name = f"preview_{uuid.uuid4().hex[:8]}.wav"
            sf.write(out_path / file_name, audio_data, sr)
            print(f"PROGRESS:{i+1}/{args.num_samples}")

    except Exception as e:
        print(f"Generation error: {e}")

if __name__ == "__main__":
    main()
