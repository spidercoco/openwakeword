import torch
from torch import optim, nn
import torchinfo
import torchmetrics
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
from tqdm import tqdm
import yaml
from pathlib import Path
import openwakeword
from openwakeword.data import generate_adversarial_texts, augment_clips, mmap_batch_generator
from openwakeword.utils import compute_features_from_generator
from openwakeword.utils import AudioFeatures

config = yaml.load(open(args.training_config, 'r').read(), yaml.Loader)

# imports Piper for synthetic sample generation
sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))
from generate_samples import generate_samples

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

# Generate positive clips for training
logging.info("#"*50 + "\nGenerating positive clips for training\n" + "#"*50)
if not os.path.exists(positive_train_output_dir):
    os.mkdir(positive_train_output_dir)
n_current_samples = len(os.listdir(positive_train_output_dir))
if n_current_samples <= 0.95*config["n_samples"]:
    generate_samples(
        text=config["target_phrase"], max_samples=config["n_samples"]-n_current_samples,
        batch_size=config["tts_batch_size"],
        noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[0.75, 1.0, 1.25],
        output_dir=positive_train_output_dir, auto_reduce_batch_size=True,
        file_names=[uuid.uuid4().hex + ".wav" for i in range(config["n_samples"])]
    )
    torch.cuda.empty_cache()
else:
    logging.warning(f"Skipping generation of positive clips for training, as ~{config['n_samples']} already exist")

# Generate positive clips for testing
logging.info("#"*50 + "\nGenerating positive clips for testing\n" + "#"*50)
if not os.path.exists(positive_test_output_dir):
    os.mkdir(positive_test_output_dir)
n_current_samples = len(os.listdir(positive_test_output_dir))
if n_current_samples <= 0.95*config["n_samples_val"]:
    generate_samples(text=config["target_phrase"], max_samples=config["n_samples_val"]-n_current_samples,
                     batch_size=config["tts_batch_size"],
                     noise_scales=[1.0], noise_scale_ws=[1.0], length_scales=[0.75, 1.0, 1.25],
                     output_dir=positive_test_output_dir, auto_reduce_batch_size=True)
    torch.cuda.empty_cache()
else:
    logging.warning(f"Skipping generation of positive clips testing, as ~{config['n_samples_val']} already exist")

# Generate adversarial negative clips for training
logging.info("#"*50 + "\nGenerating negative clips for training\n" + "#"*50)
if not os.path.exists(negative_train_output_dir):
    os.mkdir(negative_train_output_dir)
n_current_samples = len(os.listdir(negative_train_output_dir))
if n_current_samples <= 0.95*config["n_samples"]:
    adversarial_texts = config["custom_negative_phrases"]
    for target_phrase in config["target_phrase"]:
        adversarial_texts.extend(generate_adversarial_texts(
            input_text=target_phrase,
            N=config["n_samples"]//len(config["target_phrase"]),
            include_partial_phrase=1.0,
            include_input_words=0.2))
    generate_samples(text=adversarial_texts, max_samples=config["n_samples"]-n_current_samples,
                     batch_size=config["tts_batch_size"]//7,
                     noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[0.75, 1.0, 1.25],
                     output_dir=negative_train_output_dir, auto_reduce_batch_size=True,
                     file_names=[uuid.uuid4().hex + ".wav" for i in range(config["n_samples"])]
                     )
    torch.cuda.empty_cache()
else:
    logging.warning(f"Skipping generation of negative clips for training, as ~{config['n_samples']} already exist")

# Generate adversarial negative clips for testing
logging.info("#"*50 + "\nGenerating negative clips for testing\n" + "#"*50)
if not os.path.exists(negative_test_output_dir):
    os.mkdir(negative_test_output_dir)
n_current_samples = len(os.listdir(negative_test_output_dir))
if n_current_samples <= 0.95*config["n_samples_val"]:
    adversarial_texts = config["custom_negative_phrases"]
    for target_phrase in config["target_phrase"]:
        adversarial_texts.extend(generate_adversarial_texts(
            input_text=target_phrase,
            N=config["n_samples_val"]//len(config["target_phrase"]),
            include_partial_phrase=1.0,
            include_input_words=0.2))
    generate_samples(text=adversarial_texts, max_samples=config["n_samples_val"]-n_current_samples,
                     batch_size=config["tts_batch_size"]//7,
                     noise_scales=[1.0], noise_scale_ws=[1.0], length_scales=[0.75, 1.0, 1.25],
                     output_dir=negative_test_output_dir, auto_reduce_batch_size=True)
    torch.cuda.empty_cache()
else:
    logging.warning(f"Skipping generation of negative clips for testing, as ~{config['n_samples_val']} already exist")
