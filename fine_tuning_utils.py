import transformers
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM, TrainingArguments, Trainer
from datasets import load_dataset, load_metric, ClassLabel, Audio, Dataset
import random
import pandas as pd
import math
import numpy as np
import librosa
import os
import torch
from pydub import AudioSegment
from IPython.display import display, HTML
import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# https://huggingface.co/transformers/main_classes/logging.html
# verbosity set to print errors only, by default it is set to 30 = error and warnings
# transformers.logging.set_verbosity(40)


def dataset_to_csv(dataset_dir: str, output_file: str):
    wavfile_data = []
    textfile_data = []
    for (root, dirs, files) in os.walk(dataset_dir, topdown=True):
        for fn in files:
            if fn.endswith(".wav"):
                wav_id = os.path.splitext(fn)[0]
                path = os.path.join(root, fn)
                wavfile_data.append((wav_id, fn, path))
            elif fn.endswith(".txt-utf8"):
                text_id = os.path.splitext(fn)[0]
                with open(os.path.join(root, fn)) as text_file:
                    text = text_file.read()
                textfile_data.append((text_id, text))
    df_wav = pd.DataFrame(wavfile_data, columns=["segment_id", "wav_file", "path"])
    df_wav = df_wav.set_index("segment_id")
    df_text = pd.DataFrame(textfile_data, columns=["segment_id", "text"])
    df_text = df_text.set_index("segment_id")
    df_final = df_wav.merge(df_text, left_index=True, right_index=True)
    df_final.to_csv(output_file)


def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


def load_dataset_from_csv(data_files: list[str]):
    # load dataset from csv files
    dataset = load_dataset("csv", data_files=data_files)
    # split dataset
    dataset = dataset["train"]
    dataset = dataset.train_test_split(test_size=0.1)
    # loading audio
    dataset = dataset.cast_column("path", Audio())
    dataset = dataset.rename_column("path", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    # preprocess dataset
    dataset = dataset.map(prepare_dataset,
                          remove_columns=dataset.column_names["train"],
                          num_proc=4)
    return dataset


def load_dataset_from_df(data_dir_list: list[str]):
    frames = []
    for path in data_dir_list:
        wavfile_data = []
        textfile_data = []
        for (root, dirs, files) in os.walk(path, topdown=True):
            for fn in files:
                if fn.endswith(".wav"):
                    wav_id = os.path.splitext(fn)[0]
                    path = os.path.join(root, fn)
                    wavfile_data.append((wav_id, fn, path))
                elif fn.endswith(".txt-utf8"):
                    text_id = os.path.splitext(fn)[0]
                    with open(os.path.join(root, fn), encoding="utf-8-sig") as text_file:
                        text = text_file.read()
                    textfile_data.append((text_id, text))
        df_wav = pd.DataFrame(wavfile_data, columns=["segment_id", "wav_file", "path"])
        df_wav = df_wav.set_index("segment_id")
        df_text = pd.DataFrame(textfile_data, columns=["segment_id", "text"])
        df_text = df_text.set_index("segment_id")
        dataset_df = df_wav.merge(df_text, left_index=True, right_index=True)
        frames.append(dataset_df)
    # concat to full dataframe
    full_dataset_df = pd.concat(frames)
    dataset = Dataset.from_pandas(full_dataset_df)
    # split dataset
    dataset = dataset.train_test_split(test_size=0.1)
    # loading audio
    dataset = dataset.cast_column("path", Audio())
    dataset = dataset.rename_column("path", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    # preprocess dataset
    dataset = dataset.map(prepare_dataset,
                          remove_columns=dataset.column_names["train"],
                          num_proc=4)
    return dataset
