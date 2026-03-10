# 基于 OpenWakeWord 与 Qwen3-TTS 训练自定义唤醒词实战

在开发基于千问的家庭智能音箱时，唤醒词方案是核心环节。目前主流方案包括 Picovoice 的 **Porcupine**（离线、支持 Java）、**科大讯飞**以及 **OpenWakeWord**。

由于现有系统基于 Java 与云端接口，在尝试多种方案后，我决定基于 OpenWakeWord 进行自定义训练。参考了官方 Notebook 和网上诸多教程，但由于 Python 库版本兼容性问题屡屡失败。最终，我通过手动修正脚本并借助大模型修复了所有坑点。以下是详细的过程记录，希望能帮助到有需要的同学。

---

### 1. 生成正向样本 (Positive Samples)

OpenWakeWord 官方使用的是 [synthetic_speech_dataset_generation](https://github.com/dscripka/synthetic_speech_dataset_generation)。

> **注意**：建议使用 Python 3.9 安装，版本过高会导致严重的依赖冲突。

```bash
git clone https://github.com/dscripka/synthetic_speech_dataset_generation
pip install -r requirements.txt
```

#### 使用 Qwen3-TTS 生成
这里我们尝试使用 **Qwen3-TTS** 来生成更高质量的合成语音。

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```
TTS按官网的建议我们使用的是python3.12，后续正负样本的生成和训练用的是python3.10。

#### 采样率坑点 (Sampling Rate)
Qwen3-TTS 默认生成的音频采样率是 **24kHz**，而 OpenWakeWord这套训练脚本预期的是 **16kHz**。这会导致后期混合逻辑中出现张量维度不匹配的错误。必须通过脚本将所有 `.wav` 统一重采样。

#### 幻觉问题 (Hallucination)
在使用 `Qwen3-TTS-12Hz-1.7B-CustomVoice` 生成 5000 条样本时，发现数量增多后模型会出现“幻觉”（胡言乱语，内容与 Text 不符）。
**优化对策：**
1.  **音色过滤**：仅保留中文音色，剔除方言。
2.  **指令控制**：降低 `instruct` 比例（80% 标准 + 20% 指令）。
3.  **对齐优化**：强制提取音频最后 **1.3s**，因为唤醒词位于句末，这样能有效过滤掉前面的幻觉内容。

** Speaker 配置 (Chinese) **
| Speaker | Voice Description | Native language |
| :--- | :--- | :--- |
| **Vivian** | Bright, slightly edgy young female voice. | Chinese |
| **Serena** | Warm, gentle young female voice. | Chinese |
| **Uncle_Fu** | Seasoned male voice with a low, mellow timbre. | Chinese |
| **Dylan** | Youthful Beijing male voice with a clear, natural timbre. | Chinese (Beijing Dialect) |

---

## 2. 负样本数据准备 (Negative Samples)

### 外部数据集下载
- **FMA**: 下载 [fma-large sample](https://github.com/dscripka/openWakeWord#datasets) 并解压。
- **FSD50k**: 下载 [FSD50k sample](https://github.com/dscripka/openWakeWord#datasets) 并解压。
- **Common Voice**: 从 [Mozilla Common Voice](https://datacollective.mozillafoundation.org/datasets/cmj8u3q2n00vhnxxbzrjcugwc) 下载中文数据集。


#### Common Voice

Common Voice (Speech)：这是最关键的负样本，它让模型学会识别并拒绝除了唤醒词以外的所有人声。

Git上提供的下载地址已经失效了，这里我用的是[Mozilla Common Voice](https://datacollective.mozillafoundation.org/datasets/cmj8u3q2n00vhnxxbzrjcugwc)。这个数据集非常大，有20多G。这里我们只会从中
使用自定义脚本提取数据：
```bash
python extract_cv_zh.py
```



---

## 3. 环境配置与依赖修复

建议使用 **Python 3.10**：

```bash
# 安装指定版本的 Torch 以保证兼容性
pip install torch==1.12.1 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html

# 限制 Numpy 版本防止 API 冲突
pip install "numpy<2"
pip install datasets==2.18.0
```

### 补充模型资源
如果缺少 ONNX 模型文件，需手动下载至 OpenWakeWord 资源目录：
```bash
# 替换为你的实际路径
LIB_PATH="/Users/jingyehuang/miniconda3/envs/python310/lib/python3.10/site-packages/openwakeword/resources/models/"

curl -L https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -o ${LIB_PATH}melspectrogram.onnx
curl -L https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -o ${LIB_PATH}embedding_model.onnx
```

---

## 4. 关键问题修复 (Troubleshooting)

### 问题一：Torchaudio `backend` 报错
**报错信息：**
```python
TypeError: load() got an unexpected keyword argument 'backend'
```
**原因**：SpeechBrain 1.0+ 调用了旧版 Torchaudio 不存在的参数。
**解决**：将 `speechbrain` 降级至 **0.5.15**。

### 问题二：Numpy 维度参数错误
**报错位置** (`openwakeword/data.py` Line 466)：
```python
error_index = torch.from_numpy(np.where(mixed_clips_batch.max(dim=1) != 0)[0])
```
**原因**：代码在将张量转为 Numpy 后仍使用 Torch 的 `dim` 参数。
**解决**：将 `dim=1` 修改为 `axis=1`。

```bash
sed -i '' '466s/max(dim=1)/max(axis=1)/' [你的库路径]/openwakeword/data.py
```

### 问题三：24k 采样率导致对齐失败与越界
**报错信息：**
```python
RuntimeError: The size of tensor a (20750) must match the size of tensor b (30720) at non-singleton dimension 0
```
**原因**：
1. Qwen3-TTS 默认生成 24kHz 音频，而 `positive.py` 脚本硬编码 `sr = 16000`。
2. 在计算 `starts` 索引时，脚本按 16k 预估样本数。实际加载 24k 音频时，前台音频（fg）的样本数远超预期（例如 1.3s 音频在 24k 下有 31200 样本，在 16k 下仅 20800）。
3. 这种“大 fg”混合到“小 bg”中，直接导致了索引越界报错。
**解决**：必须使用 `resample.py` 将所有合成音频强制重采样为 **16kHz 单声道 (Mono)**。

### 问题四：Qwen3-TTS 幻觉与对齐优化
**问题描述**：Recall 召回率低，合成音频经常在“小熊”之后开始长篇大论。
**优化逻辑**：
1. **尾部提取 (Tail Extraction)**：由于我们统一将唤醒词放在句末（如“你好小熊”），在 `generate_qwen3_clips.py` 中强制只截取音频的最后 **1.3 秒**。这样即使模型在开头有幻觉，我们提取的也是干净的唤醒词结尾。
2. **文本变体**：混合使用“小熊”、“小熊小熊”、“你好小熊”、“嘿小熊”四种变体，提升模型对不同语境的适应力。
3. **混合比率**：设置 `INSTRUCT_PROB = 0.2`。80% 使用无指令的标准语气（保证基准稳定性），20% 随机加入情感指令（提升鲁棒性）。
4. **命名规范**：带指令生成的样本以 `instruct_` 开头，标准样本以 `std_` 开头，方便后续分析与过滤。
