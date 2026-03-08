import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from pathlib import Path
from tqdm import tqdm
import openwakeword
import openwakeword.data
import openwakeword.utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_input_dir", type=str, required=True)
    parser.add_argument("--negative_input_dirs", nargs="+", default=["../../fma_sample", "../../fsd50k_sample", "../../cv_zh_test_clips"])
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    # 路径转换
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    abs_neg_dirs = [os.path.join(base_path, d.replace("../../", "")) for d in args.negative_input_dirs]

    negative_clips, _ = openwakeword.data.filter_audio_paths(
        abs_neg_dirs,
        min_length_secs = 1.0,
        max_length_secs = 60*30,
        duration_method = "header"
    )

    positive_clips, durations = openwakeword.data.filter_audio_paths(
        [args.positive_input_dir],
        min_length_secs = 0.3,
        max_length_secs = 2.0, # 保持较宽松的长度限制以兼容 Qwen3 输出
        duration_method = "header"
    )

    N_total = len(positive_clips)
    if N_total == 0:
        print(f"Error: No valid positive clips found in {args.positive_input_dir}")
        return

    sr = 16000
    total_length_seconds = 3
    total_length = int(sr*total_length_seconds)

    jitters = (np.random.uniform(0, 0.2, len(positive_clips))*sr).astype(np.int32)
    starts = [total_length - (int(np.ceil(i*sr))+j) for i,j in zip(durations, jitters)]

    batch_size = 8
    mixing_generator = openwakeword.data.mix_clips_batch(
        foreground_clips = positive_clips,
        background_clips = negative_clips,
        combined_size = total_length,
        batch_size = batch_size,
        snr_low = 5,
        snr_high = 15,
        start_index = starts,
        volume_augmentation=True,
    )

    mixed_clips, labels, background_clips = next(mixing_generator)

    F = openwakeword.utils.AudioFeatures()

    N_total = len(positive_clips) # maximum number of rows in mmap file

    n_feature_cols = F.get_embedding_shape(total_length_seconds)

    fp = open_memmap(args.output_file, mode='w+', dtype=np.float32, shape=(N_total, n_feature_cols[0], n_feature_cols[1]))

    row_counter = 0
    # 使用直接迭代，避免 next() 导致的 StopIteration
    for batch in tqdm(mixing_generator, total=N_total//batch_size):
        batch, lbls, background = batch[0], batch[1], batch[2]
        features = F.embed_clips(batch, batch_size=256)
        
        actual_batch_size = min(features.shape[0], N_total - row_counter)
        if actual_batch_size <= 0: break
        
        fp[row_counter:row_counter+features.shape[0], :, :] = features
        row_counter += features.shape[0]
        fp.flush()
        
        # Web 进度显示
        print(f"PROGRESS:{row_counter}/{N_total}", flush=True)
        if row_counter >= N_total: break

    openwakeword.data.trim_mmap(args.output_file)
    print(f"Finished: {row_counter} samples processed.")

if __name__ == "__main__":
    main()
