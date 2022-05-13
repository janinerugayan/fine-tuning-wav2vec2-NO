import transformers
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM, TrainingArguments, Trainer
from datasets import load_dataset, load_metric, ClassLabel, Audio
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
transformers.logging.set_verbosity(40)





# ---------------------------------------------------
# DATASET PREPARATION
# ---------------------------------------------------

# Stortinget
dataset_Stortinget = "../../datasets/NordTrans_TUL/Stortinget"
wavfile_data = []
textfile_data = []
for (root, dirs, files) in os.walk(dataset_Stortinget, topdown=True):
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
df_Stortinget = df_wav.merge(df_text, left_index=True, right_index=True)
df_Stortinget.to_csv("Stortinget_TUL_train.csv")

# NRK
# dataset_NRK = "../../datasets/NordTrans_TUL/NRK"
# wavfile_data = []
# textfile_data = []
# for (root, dirs, files) in os.walk(dataset_NRK, topdown=True):
#     for fn in files:
#         if fn.endswith(".wav"):
#             wav_id = os.path.splitext(fn)[0]
#             path = os.path.join(root, fn)
#             wavfile_data.append((wav_id, fn, path))
#         elif fn.endswith(".txt-utf8"):
#             text_id = os.path.splitext(fn)[0]
#             with open(os.path.join(root, fn)) as text_file:
#                 text = text_file.read()
#             textfile_data.append((text_id, text))
# df_wav = pd.DataFrame(wavfile_data, columns=["segment_id", "wav_file", "path"])
# df_wav = df_wav.set_index("segment_id")
# df_text = pd.DataFrame(textfile_data, columns=["segment_id", "text"])
# df_text = df_text.set_index("segment_id")
# df_NRK = df_wav.merge(df_text, left_index=True, right_index=True)
# df_NRK.to_csv("NRK_TUL_train.csv")





# ---------------------------------------------------
# LOAD DATASET FROM CSV FILES
# ---------------------------------------------------

# loading dataset
data_files = ["Stortinget_TUL_train.csv"]
stortinget_dataset = load_dataset("csv", data_files=data_files)

# split dataset
stortinget_dataset = stortinget_dataset["train"]
stortinget_dataset = stortinget_dataset.train_test_split(test_size=0.1)
print(stortinget_dataset)

# loading audio
stortinget_dataset = stortinget_dataset.cast_column("path", Audio())
stortinget_dataset = stortinget_dataset.rename_column("path", "audio")
stortinget_dataset = stortinget_dataset.cast_column("audio", Audio(sampling_rate=16_000))
print(stortinget_dataset["train"][0])

# example:
rand_int = random.randint(0, len(stortinget_dataset))
print("Target text:", stortinget_dataset["train"][rand_int]["text"])
print("Input array shape:", np.asarray(stortinget_dataset["train"][rand_int]["audio"]["array"]).shape)
print("Sampling rate:", stortinget_dataset["train"][rand_int]["audio"]["sampling_rate"])

# load pretrained model
model_name = 'NbAiLab/nb-wav2vec2-1b-bokmaal'
processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)
# model = Wav2Vec2ForCTC.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

# preprocess dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

stortinget_dataset = stortinget_dataset.map(prepare_dataset, remove_columns=stortinget_dataset.column_names["train"], num_proc=4)





# ---------------------------------------------------
# SET-UP TRAINER
# ---------------------------------------------------

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# feature extraction does not need further fine-tuning
model.freeze_feature_encoder()

# training arguments
training_args = TrainingArguments(
  output_dir="../../fine-tuning_wav2vec2",
  group_by_length=True,
  per_device_train_batch_size=32,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  gradient_checkpointing=True,
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=stortinget_dataset["train"],
    eval_dataset=stortinget_dataset["test"],
    tokenizer=processor.feature_extractor,
)





# ---------------------------------------------------
# TRAINING
# ---------------------------------------------------

trainer.train()
