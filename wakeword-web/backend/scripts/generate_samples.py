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
    print("Error: Qwen3TTSModel not found. Ensure the Qwen3-TTS repo is in your PYTHONPATH.")
    exit(1)

# ================= 配置 =================
MODEL_PATH = "/data/model/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# 预设音色
PRESET_SPEAKERS = ["vivian", "serena", "uncle_fu", "dylan"]

# 语速指令映射
SPEED_STYLES = [
    ("慢速", "语速慢一点地说话"),
    ("正常", "正常语速说话"),
    ("快速", "语速快一点地说话")
]

def main():
    parser = argparse.ArgumentParser(description="Generate wake word samples with random speakers and specific speeds")
    parser.add_argument("--wakeword", type=str, required=True, help="The exact wake word to generate")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated samples")
    parser.add_argument("--num_samples", type=int, default=3, help="Total samples to generate")
    parser.add_argument("--speaker", type=str, default=None, help="Ignored, will use random from presets")
    
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 加载模型
    dtype = torch.bfloat16 if "cuda" in DEVICE else (torch.float16 if "mps" in DEVICE else torch.float32)
    print(f"Loading Qwen3-TTS on {DEVICE}...")
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=dtype, trust_remote_code=True)

    print(f"Generating samples with random speakers and rotating speeds for: {args.wakeword}")

    count = 0
    while count < args.num_samples:
        # 语速循环
        style_name, speed_instruct = SPEED_STYLES[count % len(SPEED_STYLES)]
        # 音色随机
        speaker = random.choice(PRESET_SPEAKERS)
        
        try:
            # 合成音频
            wavs, sr = model.generate_custom_voice(
                text=[args.wakeword],
                language=["Chinese"],
                speaker=[speaker],
                instruct=[speed_instruct]
            )

            audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]
            
            # 文件名包含语速和音色标识
            clean_text = args.wakeword.replace("，", "").replace(" ", "").replace("?", "")
            file_id = f"spd_{style_name}_spk_{speaker}_{clean_text}_{uuid.uuid4().hex[:6]}"
            file_name = f"{file_id}.wav"
            
            sf.write(out_path / file_name, audio_data, sr)
            count += 1
            
            print(f"PROGRESS:{count}/{args.num_samples}")
            print(f"[{count}/{args.num_samples}] Saved: {file_name} (Style: {style_name}, Speaker: {speaker})", flush=True)

        except Exception as e:
            print(f"Generation error: {e}")
            continue

    print(f"Success. Generated {count} samples in {args.output_dir}")

if __name__ == "__main__":
    main()
