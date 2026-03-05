import os
import pandas as pd
import numpy as np
import scipy.io.wavfile
import librosa
from tqdm import tqdm

# ================= 配置 =================
# 1. 解压后的主目录 (根据你之前的 ls 结果)
BASE_DIR = "cv-corpus-24.0-2025-12-05/zh-CN"
# 2. 索引文件和音频文件夹路径
TSV_PATH = os.path.join(BASE_DIR, "test.tsv")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
# 3. 输出目录
OUTPUT_DIR = "cv_zh_test_clips"
# 4. 提取数量
LIMIT = 5000
# ========================================

def main():
    # 检查路径是否存在
    if not os.path.exists(TSV_PATH):
        print(f"错误: 找不到索引文件 {TSV_PATH}")
        return
    if not os.path.exists(CLIPS_DIR):
        print(f"错误: 找不到音频文件夹 {CLIPS_DIR}")
        return

    # 1. 读取 TSV 索引
    print(f"正在读取索引: {TSV_PATH}...")
    df = pd.read_csv(TSV_PATH, sep="\t", low_memory=False)
    
    # 只要前 LIMIT 条
    df_subset = df.head(LIMIT)
    print(f"准备转换 {len(df_subset)} 条音频...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 遍历并保存
    # 使用 librosa 处理 MP3 解码和重采样，这是最稳的方法
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="处理进度"):
        try:
            # 拼接输入和输出路径
            # Common Voice 的 path 列通常就是文件名
            input_path = os.path.join(CLIPS_DIR, row['path'])
            output_filename = os.path.splitext(row['path'])[0] + ".wav"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            # 如果已经存在，跳过（方便断点续传）
            if os.path.exists(output_path):
                continue

            # 加载并重采样
            # sr=16000 自动处理采样率, mono=True 自动处理单声道
            audio, _ = librosa.load(input_path, sr=16000, mono=True)

            # 转换为 16-bit PCM 格式
            wav_data = (audio * 32767).astype(np.int16)
            
            # 写入 WAV 文件
            scipy.io.wavfile.write(output_path, 16000, wav_data)
            
        except Exception as e:
            print(f"\n处理样本 {row['path']} 出错: {e}")
            continue

    print(f"\n提取完成！音频保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
