# 小熊唤醒词训练项目 (OpenWakeWord + Qwen3-TTS)

本项目使用 Qwen3-TTS 生成高质量合成语音，结合 OpenWakeWord 框架训练自定义唤醒词“小熊”。

## 快速开始流程

按照以下步骤从零开始训练你的模型：

### 1. 生成正样本音频 (Synthesize Audio)
使用 Qwen3-TTS 模型生成包含唤醒词的原始音频。该脚本会自动截取音频末尾的 1.3 秒，以确保唤醒词位置对齐并减少模型幻觉。

```bash
# 建议先清空旧数据
rm -rf wake_word_dataset_5000

# 运行生成脚本 (生成 5000 条样本)
python generate_qwen3_clips.py
```

### 2. 音频重采样 (Resample to 16kHz)
OpenWakeWord 训练需要 16kHz 的单声道 PCM 音频。由于 Qwen3-TTS 默认输出 24kHz，需要转换。

```bash
# 将 wavs 目录下的音频转为 16k，存入 wavs_new
python resample.py
```

### 3. 生成正样本特征 (Prepare Positive Features)
读取重采样后的音频，并将其与背景负样本（如 `cv_zh_test_clips`）进行混合，生成 `.npy` 特征文件。

```bash
python positive.py
```

### 4. 训练模型 (Training)
使用生成的正负样本特征进行模型训练，并导出为 `.pth` 和 `.onnx` 格式。

```bash
python train.py
```

---

## 核心脚本说明

| 脚本名 | 功能 | 备注 |
| :--- | :--- | :--- |
| `generate_qwen3_clips.py` | 使用 Qwen3-TTS 生成音频 | 包含尾部截断逻辑，自动过滤幻觉音频 |
| `resample.py` | 批量 24kHz -> 16kHz 转换 | 使用 ffmpeg 后台处理 |
| `positive.py` | 数据增强与特征提取 | 混合 SNR，对齐唤醒词末尾 |
| `negative.py` | 提取负样本特征 | (可选) 如果 `negative_features.npy` 不存在时运行 |
| `train.py` | 模型训练与导出 | 包含 FCN 网络结构和 ONNX 导出 |

## 环境要求
- Python 3.10+
- PyTorch 1.12.1 / Torchaudio 0.12.1
- SpeechBrain 0.5.15 (降级版以保证兼容性)
- openwakeword 0.6.0
- FFmpeg (用于 resample.py)

## 常见问题
- **Recall 召回率低**：检查 `wake_word_dataset_5000/wavs` 中的音频是否清晰。如果幻觉严重，请再次运行 `generate_qwen3_clips.py`。
- **维度不匹配错误**：确保运行了 `resample.py` 且 `positive.py` 指向的是 `wavs_new` 目录。
