import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # or "0,1" for multiple GPUs

import collections
if not hasattr(collections, "Container"):
    import collections.abc
    collections.Container = collections.abc.Container
# import transformers
from transformers import AutoTokenizer, BertModel
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM, TrainingArguments, Trainer
from datasets import load_dataset, load_metric, ClassLabel, Audio, Dataset
import random
import pandas as pd
# import math
import numpy as np
# import librosa
import os
import torch
# from pydub import AudioSegment
# from IPython.display import display, HTML
import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import wandb
import argparse
# import types
from customCTCwithASD_v2 import *
import sys
import time
from dtw import *


# https://huggingface.co/transformers/main_classes/logging.html
# verbosity set to print errors only, by default it is set to 30 = error and warnings
# transformers.logging.set_verbosity(40)

# enabled to find the operation that failed to compute its gradient
# torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def compute_asd_score(model=None, tokenizer=None, reference=None, hypothesis=None):
        asd_score = 0
        num_ref_hyp_pairs = 0
        for ref_text, hyp_text in zip(reference, hypothesis):
            ref_text = re.sub(r"\s+", " ", ref_text.replace("[UNK]", ""))  # removes the [UNK] token in the reference text, observed during training
            hyp_text = re.sub(r"\s+", " ", hyp_text.replace("[UNK]", ""))
            tokenized_ref = tokenizer(ref_text.lower(), padding=True, truncation=True, max_length=512, return_tensors="pt")
            tokenized_hyp = tokenizer(hyp_text.lower(), padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                model_output_ref = model(**tokenized_ref, output_hidden_states=True)
                model_output_hyp = model(**tokenized_hyp, output_hidden_states=True)
            hidden_states_ref = model_output_ref.hidden_states
            hidden_states_hyp = model_output_hyp.hidden_states
            all_layers_reference = [hidden_states_ref[1].squeeze(), hidden_states_ref[2].squeeze(), hidden_states_ref[3].squeeze(), hidden_states_ref[4].squeeze(),
                                    hidden_states_ref[5].squeeze(), hidden_states_ref[6].squeeze(), hidden_states_ref[7].squeeze(), hidden_states_ref[8].squeeze(),
                                    hidden_states_ref[9].squeeze(), hidden_states_ref[10].squeeze(), hidden_states_ref[11].squeeze(), hidden_states_ref[12].squeeze()]
            all_layers_hypothesis = [hidden_states_hyp[1].squeeze(), hidden_states_hyp[2].squeeze(), hidden_states_hyp[3].squeeze(), hidden_states_hyp[4].squeeze(),
                                     hidden_states_hyp[5].squeeze(), hidden_states_hyp[6].squeeze(), hidden_states_hyp[7].squeeze(), hidden_states_hyp[8].squeeze(),
                                     hidden_states_hyp[9].squeeze(), hidden_states_hyp[10].squeeze(), hidden_states_hyp[11].squeeze(), hidden_states_hyp[12].squeeze()]
            output_mean_reference = torch.stack(all_layers_reference).mean(dim=0)
            output_mean_hypothesis = torch.stack(all_layers_hypothesis).mean(dim=0)
            alignment = dtw(output_mean_hypothesis, output_mean_reference, dist_method=distance.cosine, keep_internals=True)
            num_tokens = len(output_mean_reference)
            # min_global_distance_norm = (alignment.distance / num_tokens)
            asd_score += (alignment.distance / num_tokens)
            num_ref_hyp_pairs += 1
        # return min_global_distance_norm
        return asd_score / num_ref_hyp_pairs


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\*]'
def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    print(df)


def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


def load_dataset_from_files(data_dir_list:list[str], csv_export_dir:str, split_ratio=0.1, csv_export=True):
    frames = []
    for path in data_dir_list:
        source = os.path.basename(os.path.dirname(path))
        wavfile_data = []
        textfile_data = []
        for (root, dirs, files) in os.walk(path, topdown=True):
            if source == "Rundkast":  # to modify depending on Rundkast cuts folder name
                for fn in files:
                    if fn.endswith(".wav"):
                        wav_id = source + "_" + os.path.splitext(fn)[0]
                        path = os.path.join(root, fn)
                        wavfile_data.append((wav_id, fn, path, source))
                    elif fn.endswith(".txt"):
                        text_id = source + "_" + os.path.splitext(fn)[0]
                        with open(os.path.join(root, fn), encoding="utf-8") as text_file:
                            text = text_file.read()
                        textfile_data.append((text_id, text))
            else:
                for fn in files:
                    if fn.endswith(".wav"):
                        wav_id = source + "_" + os.path.splitext(fn)[0]
                        path = os.path.join(root, fn)
                        wavfile_data.append((wav_id, fn, path, source))
                    elif fn.endswith(".txt-utf8"):
                        text_id = source + "_" + os.path.splitext(fn)[0]
                        with open(os.path.join(root, fn), encoding="utf-8-sig") as text_file:
                            text = text_file.read()
                        textfile_data.append((text_id, text))
        df_wav = pd.DataFrame(wavfile_data, columns=["segment_id", "wav_file", "path", "source"])
        df_wav = df_wav.set_index("segment_id")
        df_text = pd.DataFrame(textfile_data, columns=["segment_id", "text"])
        df_text = df_text.set_index("segment_id")
        dataset_df = df_wav.merge(df_text, left_index=True, right_index=True)
        frames.append(dataset_df)
    # concat to full dataframe and convert to Dataset with special characters removed
    full_dataset_df = pd.concat(frames)
    raw_dataset = Dataset.from_pandas(full_dataset_df)
    raw_dataset = raw_dataset.map(remove_special_characters)
    # split dataset
    raw_dataset = raw_dataset.train_test_split(test_size=split_ratio, seed=42)
    # save copy of dataset
    if csv_export is True:
        df_train = pd.DataFrame(raw_dataset["train"])
        df_train.to_csv(os.path.join(csv_export_dir, "train_set.csv"))
        df_dev = pd.DataFrame(raw_dataset["test"])
        df_dev.to_csv(os.path.join(csv_export_dir, "dev_set.csv"))
    # loading audio
    dataset = raw_dataset.cast_column("path", Audio())
    dataset = dataset.rename_column("path", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    # preprocess dataset
    dataset = dataset.map(prepare_dataset,
                          remove_columns=dataset.column_names["train"],
                          num_proc=4)
    return raw_dataset, dataset




# ---------------------------------------------------
# ARGUMENTS & INTIALIZATIONS
# ---------------------------------------------------

# parser=argparse.ArgumentParser()
# parser.add_argument("--original_model",         type=str)
# parser.add_argument("--fine_tuned_model_ver",   type=str)
# parser.add_argument("--export_model_dir",       type=str)
# parser.add_argument("--num_train_epochs",       type=int)
# parser.add_argument("--learning_rate",          type=float)
# parser.add_argument("--lambda_asd",             type=float)
# # parser.add_argument("--lambda_ctc",             type=float)
# parser.add_argument("--use_asd_metric",         type=int)
# parser.add_argument("--wandb_name",             type=str)
# parser.add_argument("--export_log",             type=str)

parser=argparse.ArgumentParser()
parser.add_argument("--original_model",         type=str)
parser.add_argument("--fine_tuned_model_ver",   type=str)
parser.add_argument("--export_model_dir",       type=str)
parser.add_argument("--num_train_epochs",       type=int)
parser.add_argument("--learning_rate",          type=float)
parser.add_argument("--lambda_asd",             type=float)
parser.add_argument("--use_asd_metric",         type=int)
parser.add_argument("--num_paths",              type=int)
parser.add_argument("--normalized_score",       type=int)
parser.add_argument("--wandb_name",             type=str)
# parser.add_argument("--train_data",             type=str)
parser.add_argument("--from_checkpoint",        type=int)
parser.add_argument("--checkpoint_path",        type=str)
parser.add_argument("--training_data",          type=str)

args = parser.parse_args()


# WANDB login / initialization
wandb.init(project="fine-tuning-wav2vec2-NO_customLoss", entity="janinerugayan", name=args.wandb_name)

# torch.multiprocessing.set_start_method('spawn')




# ---------------------------------------------------
# LOAD PRETRAINED MODEL
# ---------------------------------------------------

# print("Loading pretrained model " + args.original_model)

# model_name = args.original_model

# processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)
# processor_woLM = Wav2Vec2Processor.from_pretrained(model_name)

# model = Wav2Vec2ForCTC.from_pretrained(
#     model_name,
#     ctc_loss_reduction="mean",
#     pad_token_id=processor.tokenizer.pad_token_id,
# )
# model = model.to(device)


# NEED TO DEFINE THE PROCESSOR IF TRAINING FROM SCRATCH!!!

processor = Wav2Vec2Processor.from_pretrained("NbAiLab/nb-wav2vec2-300m-bokmaal")

model = Wav2Vec2ForCTC.from_pretrained(
    "KBLab/wav2vec2-large-voxrex",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

# feature extraction does not need further fine-tuning
model.freeze_feature_encoder()

# wandb.watch(model, log_freq=50)


# ---------------------------------------------------
# LOAD DATASET FROM CSV FILES
# ---------------------------------------------------

if args.training_data == "load_csv":
    print("Loading dataset from CSV files")
    dataset = load_dataset("csv", data_files={"train": "./dataset/train_set.csv",
                                              "dev": "./dataset/dev_set.csv"})
    # loading audio
    dataset = dataset.cast_column("path", Audio())
    dataset = dataset.rename_column("path", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    # preprocess dataset
    dataset = dataset.map(prepare_dataset,
                          remove_columns=dataset.column_names["train"],
                          num_proc=4)
    train_dataset = dataset["train"]
    test_dataset = dataset["dev"]
else:
    print("Loading dataset direct from data dir to pandas dataframe")

    # data_dir_list = ["../../datasets/NordTrans_TUL/train/Stortinget/",
    #                  "../../datasets/NordTrans_TUL/train/NRK/",
    #                  "../../datasets/NordTrans_TUL/train/Rundkast_cuts_random25per_30secmax/"]

    data_dir_list = ["../../datasets/NordTrans_TUL/train_small/Stortinget/",
                    "../../datasets/NordTrans_TUL/train_small/NRK/",
                    "../../datasets/NordTrans_TUL/train_small/Rundkast/"]

    # data_dir_list = ["../../datasets/NordTrans_TUL/train_small/Rundkast/"]

    csv_export_dir = "./model_ckpts/" + args.fine_tuned_model_ver + "/runs/"
    raw_dataset, dataset = load_dataset_from_files(data_dir_list, csv_export_dir, split_ratio=0.1, csv_export=True)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

print(dataset)




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
    # processor: Wav2Vec2ProcessorWithLM
    padding: Union[bool, str] = True  # original: True
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


# # The bare Bert Model transformer outputting raw hidden-states without any specific head on top.
# metric_modelname = 'ltg/norbert2'  # changed to latest version of NorBERT (20-Mar-2023)
# metric_model = BertModel.from_pretrained(metric_modelname)
# metric_tokenizer = AutoTokenizer.from_pretrained(metric_modelname)

# # multi-lingual LM
# # metric_modelname = "bert-base-multilingual-cased"
# # metric_model_multi = BertModel.from_pretrained(metric_modelname)
# # metric_tokenizer_multi = AutoTokenizer.from_pretrained(metric_modelname)

# asd_metric = load_metric("asd_metric.py")
# wer_metric = load_metric("wer")

# def compute_metrics(pred):
#     pred_logits = pred.predictions
#     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
#     pred_str = processor.batch_decode(pred_logits)
#     label_str = processor_woLM.batch_decode(pred.label_ids, group_tokens=False)  # we do not want to group tokens when computing the metrics
#     wer = wer_metric.compute(predictions=pred_str.text, references=label_str) # worked in fine-tuning versions 1 to 14 (wer metric)
#     # ADD ASD HERE!
#     asd = asd_metric.compute(model=metric_model, tokenizer=metric_tokenizer, reference=label_str, hypothesis=pred_str.text)
#     # asd_multi = asd_metric.compute(model=metric_model_multi, tokenizer=metric_tokenizer_multi, reference=label_str, hypothesis=pred_str.text)

#     return {"wer": wer, "asd": asd}


# The bare Bert Model transformer outputting raw hidden-states without any specific head on top.
metric_modelname = 'ltg/norbert2'  # changed to latest version of NorBERT (20-Mar-2023)
metric_model = BertModel.from_pretrained(metric_modelname)
metric_tokenizer = AutoTokenizer.from_pretrained(metric_modelname)

# COMPUTE METRICS FOR EVA[[L
wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    asd = compute_asd_score(model=metric_model, tokenizer=metric_tokenizer, reference=label_str, hypothesis=pred_str)

    return {"wer": wer, "asd": asd}




print("Available cuda devices:", torch.cuda.device_count())


repo_local_dir = "./model_ckpts/" + args.fine_tuned_model_ver + "/"
# training arguments
training_args = TrainingArguments(
  output_dir=repo_local_dir,
  group_by_length=True,
  per_device_train_batch_size=8,  # orig: 8
  per_device_eval_batch_size=8,  # orig: 8
#   gradient_accumulation_steps=4,  # not in the original source/reference
  eval_accumulation_steps=100,
  evaluation_strategy="steps",
  num_train_epochs=args.num_train_epochs,  # orig: 30
  fp16=True,  # orig: True
  save_strategy="epoch",
  gradient_checkpointing=True,
#   save_steps=300,  # for one dataset exp
#   eval_steps=300,  # for one dataset exp
#   logging_steps=300,  # for one dataset exp
  save_steps=500,  # orig: 500
  eval_steps=500,  # orig: 500
  logging_steps=500,  # orig: 500
  logging_strategy="steps",
  learning_rate=args.learning_rate,  # orig: 1e-4
  weight_decay=0.005,
  warmup_steps=2000,  # orig: 1000
#   save_total_limit=2,
  push_to_hub=False,
  seed=42,
  data_seed=42,
  report_to="wandb"
)




if args.use_asd_metric == 1:
    print("Setting up CUSTOM Trainer")

    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):

            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
            """

            outputs = model(**inputs)
            logits = outputs["logits"]

            attention_mask = torch.ones_like(inputs["input_values"], dtype=torch.long)
            input_lengths = model._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            labels = inputs["labels"]
            label_str = processor.batch_decode(labels, group_tokens=False)  # we do not want to group tokens when computing the metrics

            """
            MASD Loss
            """
            masd_loss = Seq2seqMASDLoss(sampling_method = "beam_search",
                                        candidate_paths_num = args.num_paths,
                                        reduction = "mean",
                                        normalized_score = args.normalized_score)
            nbest_log_distribution, nbest_pred = masd_loss.get_logits_for_decoding(logits, input_lengths)

            # getting the hypotheses
            hyp_list = []
            for i in range(nbest_pred.size()[0]):
                hyp_text = processor.batch_decode(nbest_pred[i])
                hyp_list.append(hyp_text)

            asd_loss = masd_loss(nbest_log_distribution,
                                 label_str,
                                 hyp_list,
                                 metric_model,
                                 metric_tokenizer)

            total_loss = (asd_loss * args.lambda_asd) + ((1 - args.lambda_asd) * outputs["loss"])

            return (total_loss, outputs) if return_outputs else total_loss


    trainer = CustomTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
    )

else:
    print("Setting up the trainer")

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
    )




print("===== MODEL TRAINING =====")

finetuned_model_dir = args.export_model_dir
log_dir = "./model_ckpts/" + args.fine_tuned_model_ver + "/runs/"

torch.cuda.empty_cache()

# TRAINING OF MODEL
if args.from_checkpoint == 0:
    trainer.train()
elif args.from_checkpoint == 1:
    print("continuing training from checkpoint")
    trainer.train(args.checkpoint_path)

log_history_fn = os.path.join(log_dir, "log_history.txt")
with open(log_history_fn, "w") as f:
    f.write(json.dumps(trainer.state.log_history))

print("Saving fine-tuned model")
model.save_pretrained(save_directory=finetuned_model_dir)
processor.save_pretrained(save_directory=finetuned_model_dir)

