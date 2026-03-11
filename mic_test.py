import os
import sys
import numpy as np
import onnxruntime as ort
import openwakeword
import openwakeword.utils
import pyaudio
import argparse

# 确保脚本能找到 beary.onnx
DEFAULT_MODEL = "beary.onnx"

def main():
    parser = argparse.ArgumentParser(description="Microphone Real-time Wake Word Detection")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to ONNX model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0 - 1.0)")
    args = parser.parse_argument_list = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return

    # 1. 初始化推理引擎
    print(f"Loading model: {args.model}")
    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name
    model_window_size = session.get_inputs()[0].shape[1]
    
    # 初始化特征提取器
    F = openwakeword.utils.AudioFeatures()

    # 2. 初始化 PyAudio
    CHUNKS_SAMPLES = 1280 # 约 80ms 的音频块
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000 # 必须是 16kHz

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNKS_SAMPLES)

    print("\n" + "="*40)
    print(" Listening... Speak your wake word!")
    print(" (Press Ctrl+C to stop)")
    print("="*40 + "\n")

    audio_buffer = np.array([], dtype=np.int16)

    try:
        while True:
            # 读取麦克风数据
            data = stream.read(CHUNKS_SAMPLES, exception_on_overflow=False)
            chunk = np.frombuffer(data, dtype=np.int16)
            audio_buffer = np.append(audio_buffer, chunk)

            # 保持 buffer 长度在合理范围 (约 3 秒数据)
            if len(audio_buffer) > 48000:
                audio_buffer = audio_buffer[-48000:]

            # 只有当 buffer 足够长时才进行特征提取
            if len(audio_buffer) < 20480:
                continue

            # 提取特征
            audio_batch = audio_buffer.reshape(1, -1)
            features = F.embed_clips(audio_batch)
            
            if features.shape[1] >= model_window_size:
                # 取最后一窗推理
                window = features[:, -model_window_size:, :]
                outputs = session.run(None, {input_name: window})
                score = float(outputs[0][0][0])

                # 终端反馈
                sys.stdout.write(f"\rConfidence Score: {score:.4f} " + (" [DETECTED]" if score > args.threshold else "           "))
                sys.stdout.flush()

                if score > args.threshold:
                    print(f"\n\n>>> [{args.model}] WAKE WORD DETECTED! <<<\n")
                    # 检测到后立即清空 buffer，防止同一个词重复识别
                    audio_buffer = np.array([], dtype=np.int16)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
