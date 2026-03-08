import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from pathlib import Path
from tqdm import tqdm
import openwakeword
import openwakeword.data
import openwakeword.utils
import datasets

def main():
    parser = argparse.ArgumentParser()
    # 路径参数化
    parser.add_argument("--input_dirs", nargs="+", default=["../../fma_sample", "../../fsd50k_sample", "../../cv_zh_test_clips"])
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    # 逻辑完全同步原始 negative.py
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    abs_input_dirs = [os.path.join(base_path, d.replace("../../", "")) for d in args.input_dirs]

    negative_clips, negative_durations = openwakeword.data.filter_audio_paths(
        abs_input_dirs,
        min_length_secs = 1.0,
        max_length_secs = 60*30,
        duration_method = "header"
    )

    F = openwakeword.utils.AudioFeatures()
    audio_dataset = datasets.Dataset.from_dict({"audio": negative_clips})
    audio_dataset = audio_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

    batch_size = 64 
    clip_size = 3  
    N_total = int(sum(negative_durations)//clip_size) 
    n_feature_cols = F.get_embedding_shape(clip_size)

    output_array_shape = (N_total, n_feature_cols[0], n_feature_cols[1])
    fp = open_memmap(args.output_file, mode='w+', dtype=np.float32, shape=output_array_shape)

    row_counter = 0
    total_rows = audio_dataset.num_rows
    for i in tqdm(np.arange(0, audio_dataset.num_rows, batch_size)):
        wav_data = [(j["array"]*32767).astype(np.int16) for j in audio_dataset[i:i+batch_size]["audio"]]
        wav_data = openwakeword.data.stack_clips(wav_data, clip_size=16000*clip_size).astype(np.int16)
        features = F.embed_clips(x=wav_data, batch_size=1024, ncpu=8)
        
        if row_counter + features.shape[0] > N_total:
            fp[row_counter:min(row_counter+features.shape[0], N_total), :, :] = features[0:N_total - row_counter, :, :]
            fp.flush()
            break
        else:
            fp[row_counter:row_counter+features.shape[0], :, :] = features
            row_counter += features.shape[0]
            fp.flush()
        
        # 仅增加这一行用于 Web 进度显示
        print(f"PROGRESS:{i + len(wav_data)}/{total_rows}", flush=True)
            
    openwakeword.data.trim_mmap(args.output_file)

if __name__ == "__main__":
    main()
