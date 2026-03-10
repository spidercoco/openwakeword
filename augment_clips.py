import torch
from torch import optim, nn
import torchinfo
import copy
import os
import sys
import tempfile
import uuid
import numpy as np
import scipy
import collections
import argparse
import logging
import soundfile as sf
import random
from tqdm import tqdm
import yaml
from pathlib import Path
import openwakeword
from openwakeword.data import augment_clips, mmap_batch_generator
from openwakeword.utils import compute_features_from_generator
from openwakeword.utils import AudioFeatures

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# ================= 原有逻辑适配 =================

config = yaml.load(open("my_model.yaml", 'r').read(), yaml.Loader)

# Define output locations
config["output_dir"] = os.path.abspath(config["output_dir"])
if not os.path.exists(config["output_dir"]):
    os.mkdir(config["output_dir"])
if not os.path.exists(os.path.join(config["output_dir"], config["model_name"])):
    os.mkdir(os.path.join(config["output_dir"], config["model_name"]))

positive_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "positive_train")
positive_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "positive_test")
negative_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_train")
negative_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_test")
feature_save_dir = os.path.join(config["output_dir"], config["model_name"])

# Get paths for impulse response and background audio files
rir_paths = [i.path for j in config["rir_paths"] for i in os.scandir(j)]
background_paths = []
if len(config["background_paths_duplication_rate"]) != len(config["background_paths"]):
    config["background_paths_duplication_rate"] = [1]*len(config["background_paths"])
for background_path, duplication_rate in zip(config["background_paths"], config["background_paths_duplication_rate"]):
    background_paths.extend([i.path for i in os.scandir(background_path)]*duplication_rate)


positive_clips_train = [str(i) for i in Path(positive_train_output_dir).glob("*.wav")]*config["augmentation_rounds"]
positive_clips_train_generator = augment_clips(positive_clips_train, total_length=config["total_length"],
                                               batch_size=config["augmentation_batch_size"],
                                               background_clip_paths=background_paths,
                                               RIR_paths=rir_paths)

positive_clips_test = [str(i) for i in Path(positive_test_output_dir).glob("*.wav")]*config["augmentation_rounds"]
positive_clips_test_generator = augment_clips(positive_clips_test, total_length=config["total_length"],
                                              batch_size=config["augmentation_batch_size"],
                                              background_clip_paths=background_paths,
                                              RIR_paths=rir_paths)

negative_clips_train = [str(i) for i in Path(negative_train_output_dir).glob("*.wav")]*config["augmentation_rounds"]
negative_clips_train_generator = augment_clips(negative_clips_train, total_length=config["total_length"],
                                               batch_size=config["augmentation_batch_size"],
                                               background_clip_paths=background_paths,
                                               RIR_paths=rir_paths)

negative_clips_test = [str(i) for i in Path(negative_test_output_dir).glob("*.wav")]*config["augmentation_rounds"]
negative_clips_test_generator = augment_clips(negative_clips_test, total_length=config["total_length"],
                                              batch_size=config["augmentation_batch_size"],
                                              background_clip_paths=background_paths,
                                              RIR_paths=rir_paths)

# Compute features and save to disk via memmapped arrays
logging.info("#"*50 + "\nComputing openwakeword features for generated samples\n" + "#"*50)
n_cpus = os.cpu_count()
if n_cpus is None:
    n_cpus = 1
else:
    n_cpus = n_cpus//2
compute_features_from_generator(positive_clips_train_generator, n_total=len(os.listdir(positive_train_output_dir)),
                                clip_duration=config["total_length"],
                                output_file=os.path.join(feature_save_dir, "positive_features_train.npy"),
                                device="gpu" if torch.cuda.is_available() else "cpu",
                                ncpu=n_cpus if not torch.cuda.is_available() else 1)

compute_features_from_generator(negative_clips_train_generator, n_total=len(os.listdir(negative_train_output_dir)),
                                clip_duration=config["total_length"],
                                output_file=os.path.join(feature_save_dir, "negative_features_train.npy"),
                                device="gpu" if torch.cuda.is_available() else "cpu",
                                ncpu=n_cpus if not torch.cuda.is_available() else 1)

compute_features_from_generator(positive_clips_test_generator, n_total=len(os.listdir(positive_test_output_dir)),
                                clip_duration=config["total_length"],
                                output_file=os.path.join(feature_save_dir, "positive_features_test.npy"),
                                device="gpu" if torch.cuda.is_available() else "cpu",
                                ncpu=n_cpus if not torch.cuda.is_available() else 1)

compute_features_from_generator(negative_clips_test_generator, n_total=len(os.listdir(negative_test_output_dir)),
                                clip_duration=config["total_length"],
                                output_file=os.path.join(feature_save_dir, "negative_features_test.npy"),
                                device="gpu" if torch.cuda.is_available() else "cpu",
                                ncpu=n_cpus if not torch.cuda.is_available() else 1)
