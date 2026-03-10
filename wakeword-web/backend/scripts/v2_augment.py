import torch
import os
import sys
import numpy as np
import argparse
import logging
from tqdm import tqdm
import yaml
from pathlib import Path
import openwakeword
from openwakeword.data import augment_clips
from openwakeword.utils import compute_features_from_generator

# 设置日志
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to task config.yaml")
    args = parser.parse_args()

    # 1. 加载任务配置
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    task_root = os.path.dirname(os.path.abspath(args.config))
    
    # 定义目录 (严格对应 v2_resample.py 输出的目录)
    positive_train_dir = os.path.join(task_root, "positive_train")
    positive_test_dir = os.path.join(task_root, "positive_test")
    negative_train_dir = os.path.join(task_root, "negative_train")
    negative_test_dir = os.path.join(task_root, "negative_test")
    
    # 2. 获取 RIR 和背景音路径 (从根目录配置中获取或使用默认)
    # 注意：这里假设 RIR 和背景音在项目根目录下，我们需要处理路径
    # 为了简化，我们直接从 config 中读取，如果不存在则使用默认路径
    def get_abs_paths(paths):
        base_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(task_root))))
        abs_paths = []
        for p in paths:
            # 移除 ./ 前缀
            clean_p = p.replace("./", "")
            abs_paths.append(os.path.join(base_project_root, clean_p))
        return abs_paths

    rir_base_paths = get_abs_paths(config.get("rir_paths", ["mit_rirs"]))
    bg_base_paths = get_abs_paths(config.get("background_paths", ["audioset_16k", "fma"]))
    
    rir_paths = [i.path for j in rir_base_paths if os.path.exists(j) for i in os.scandir(j)]
    background_paths = []
    bg_dupe = config.get("background_paths_duplication_rate", [1] * len(bg_base_paths))
    for bg_path, dupe in zip(bg_base_paths, bg_dupe):
        if os.path.exists(bg_path):
            background_paths.extend([i.path for i in os.scandir(bg_path)] * dupe)

    # 3. 初始化增强生成器 (逻辑完全同步 augment_clips.py)
    total_length = config.get("total_length", 3 * 16000)
    aug_batch_size = config.get("augmentation_batch_size", 16)
    aug_rounds = config.get("augmentation_rounds", 1)

    def create_gen(input_dir):
        clips = [str(i) for i in Path(input_dir).glob("*.wav")] * aug_rounds
        return augment_clips(clips, total_length=total_length, batch_size=aug_batch_size,
                             background_clip_paths=background_paths, RIR_paths=rir_paths), len(os.listdir(input_dir))

    # 4. 执行特征提取
    n_cpus = os.cpu_count()
    n_cpus = n_cpus // 2 if n_cpus else 1
    device = "gpu" if torch.cuda.is_available() else "cpu"

    task_list = [
        (positive_train_dir, "positive_features_train.npy", "正向训练样本"),
        (negative_train_dir, "negative_features_train.npy", "负向训练样本"),
        (positive_test_dir, "positive_features_test.npy", "正向测试样本"),
        (negative_test_dir, "negative_features_test.npy", "负向测试样本")
    ]

    for input_dir, out_name, msg in task_list:
        if not os.path.exists(input_dir): continue
        print(f"\n--- {msg} 增强处理 ---")
        gen, n_total = create_gen(input_dir)
        output_file = os.path.join(task_root, out_name)
        
        # 封装 compute_features 以便支持进度输出
        # 注意：compute_features_from_generator 内部会打印 tqdm，我们这里通过包装来输出 PROGRESS
        # 简单起见，我们假设每个阶段执行完后更新一个大进度
        compute_features_from_generator(gen, n_total=n_total * aug_rounds,
                                        clip_duration=total_length,
                                        output_file=output_file,
                                        device=device, ncpu=n_cpus if device == "cpu" else 1)
        # 这里输出一个 PROGRESS 信号
        print(f"PROGRESS:1/1", flush=True) # 简单的阶段性反馈

if __name__ == "__main__":
    main()
