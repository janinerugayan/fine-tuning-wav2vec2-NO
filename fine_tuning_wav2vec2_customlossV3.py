import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # or "0,1" for multiple GPUs

import collections
if not hasattr(collections, "Container"):
    import collections.abc
    collections.Container = collections.abc.Container
import transformers
from transformers import AutoTokenizer, BertModel
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
import wandb
import argparse
import types
from customCTCwithASD import compute_CTCloss_withASD

# https://huggingface.co/transformers/main_classes/logging.html
# verbosity set to print errors only, by default it is set to 30 = error and warnings
# transformers.logging.set_verbosity(40)

# enabled to find the operation that failed to compute its gradient
# torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
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
    raw_dataset = raw_dataset.train_test_split(test_size=split_ratio)
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

parser=argparse.ArgumentParser()
parser.add_argument("--original_model",         type=str)
parser.add_argument("--fine_tuned_model_ver",   type=str)
parser.add_argument("--export_model_dir",       type=str)
parser.add_argument("--num_train_epochs",       type=int)
parser.add_argument("--learning_rate",          type=float)
parser.add_argument("--lambda_asd",             type=float)
parser.add_argument("--use_asd_metric",         type=int)
parser.add_argument("--wandb_name",             type=str)
parser.add_argument("--export_log",             type=str)
args = parser.parse_args()

# WANDB login / initialization
wandb.init(project="fine-tuning-wav2vec2-NO_customLoss", entity="janinerugayan", name=args.wandb_name)

# torch.multiprocessing.set_start_method('spawn')




# ---------------------------------------------------
# LOAD PRETRAINED MODEL
# ---------------------------------------------------

print("Loading pretrained model " + args.original_model)

model_name = args.original_model

processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)
processor_woLM = Wav2Vec2Processor.from_pretrained(model_name)

model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)
model = model.to(device)

# feature extraction does not need further fine-tuning
model.freeze_feature_encoder()




# ---------------------------------------------------
# LOAD DATASET FROM CSV FILES
# ---------------------------------------------------

print("Loading dataset direct from data dir to pandas dataframe")

# data_dir_list = ["../../datasets/NordTrans_TUL/train/Stortinget/",
#                  "../../datasets/NordTrans_TUL/train/NRK/",
#                  "../../datasets/NordTrans_TUL/train/Rundkast_cuts_random25per_30secmax/"]

data_dir_list = ["../../datasets/NordTrans_TUL/train_small/Stortinget/",
                 "../../datasets/NordTrans_TUL/train_small/NRK/",
                 "../../datasets/NordTrans_TUL/train_small/Rundkast/"]

# data_dir_list = ["../../datasets/NordTrans_TUL/train_small/Rundkast/"]

csv_export_dir = "../../model_ckpts/" + args.fine_tuned_model_ver + "/runs/"

raw_dataset, dataset = load_dataset_from_files(data_dir_list, csv_export_dir, split_ratio=0.1, csv_export=True)

print(raw_dataset)
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

    # processor: Wav2Vec2Processor
    processor: Wav2Vec2ProcessorWithLM
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


repo_local_dir = "../../model_ckpts/" + args.fine_tuned_model_ver + "/"
# training arguments
training_args = TrainingArguments(
  output_dir=repo_local_dir,
  group_by_length=True,
  per_device_train_batch_size=8,  # orig: 8
  per_device_eval_batch_size=8,  # orig: 8
  eval_accumulation_steps=100,
  evaluation_strategy="steps",
  num_train_epochs=args.num_train_epochs,  # orig: 30
  fp16=True,  # orig: True
  gradient_checkpointing=True,
  save_steps=500,  # orig: 500
  eval_steps=500,  # orig: 500
  logging_steps=500,  # orig: 500
  learning_rate=args.learning_rate,  # orig: 1e-4
  weight_decay=0.005,
  warmup_steps=2000,  # orig: 1000
  save_total_limit=2,
  push_to_hub=False,
  seed=42,
  data_seed=42,
  report_to="wandb"
)

# The bare Bert Model transformer outputting raw hidden-states without any specific head on top.
metric_modelname = 'ltg/norbert2'  # changed to latest version of NorBERT (20-Mar-2023)
metric_model = BertModel.from_pretrained(metric_modelname)
metric_tokenizer = AutoTokenizer.from_pretrained(metric_modelname)
asd_metric = load_metric("asd_metric.py")
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_logits)
    label_str = processor_woLM.batch_decode(pred.label_ids, group_tokens=False)  # we do not want to group tokens when computing the metrics
    wer = wer_metric.compute(predictions=pred_str.text, references=label_str) # worked in fine-tuning versions 1 to 14 (wer metric)
    # ADD ASD HERE!
    asd = asd_metric.compute(model=metric_model, tokenizer=metric_tokenizer, reference=label_str, hypothesis=pred_str.text)
    return {"wer": wer, "asd": asd}

print("Available cuda devices:", torch.cuda.device_count())

if args.use_asd_metric == 1:
    print("Setting up Custom Trainer")

    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):

            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
            """

            outputs = model(**inputs)

            attention_mask = inputs["attention_mask"]
            # print("attention mask shape:", attention_mask.shape)
            input_lengths = model._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            # print("input lengths:", input_lengths)

            output_logits = outputs["logits"]
            pred_logits = self._gather_and_numpify(output_logits.detach(), "eval_preds")
            pred_str = processor.batch_decode(pred_logits)
            labels = inputs["labels"]
            label_str = processor_woLM.batch_decode(labels, group_tokens=False)  # we do not want to group tokens when computing the metrics

            # print("REF:", label_str)
            # print("HYP:", pred_str.text)

            # batched input with only 1 GPU used
            asd_loss = compute_CTCloss_withASD(reference_text=label_str,
                                                predicted_text=pred_str.text,
                                                ref_label_ids=labels,
                                                output_logits=output_logits,
                                                input_lengths=input_lengths,
                                                asd_model=metric_model,
                                                asd_tokenizer=metric_tokenizer,
                                                lambda_asd=args.lambda_asd)

            return (asd_loss, outputs) if return_outputs else asd_loss

            # # batched input with 2 GPUs used
            # for i in range(torch.cuda.device_count()):
            #     predicted_text = pred_str.text[i*8:(i+1)*8]
            #     reference_text = label_str[i*8:(i+1)*8]
            #     logits = output_logits[i*8:(i+1)*8]
            #     label_ids = labels[i*8:(i+1)*8]
            #     if i == 0:
            #         asd_loss_batch1 = compute_CTCloss_withASD(reference_text=reference_text,
            #                                                   predicted_text=predicted_text,
            #                                                   ref_label_ids=label_ids,
            #                                                   output_logits=logits,
            #                                                   input_lengths=input_lengths)
            #                                                 #   asd_model=metric_model,
            #                                                 #   asd_tokenizer=metric_tokenizer)
            #     else:
            #         asd_loss_batch2 = compute_CTCloss_withASD(reference_text=reference_text,
            #                                                   predicted_text=predicted_text,
            #                                                   ref_label_ids=label_ids,
            #                                                   output_logits=logits,
            #                                                   input_lengths=input_lengths)
            #                                                 #   asd_model=metric_model,
            #                                                 #   asd_tokenizer=metric_tokenizer)

            # loss = torch.cat(((asd_loss_batch1).reshape(1), (asd_loss_batch2).reshape(1)), dim=0)
            # print("2 devices loss:", loss)
            # return (loss, outputs) if return_outputs else loss

            # # 1 example per batch
            # loss = compute_CTCloss_withASD(reference_text=[label_str[0]],
            #                                                 predicted_text=[pred_str.text[0]],
            #                                                 ref_label_ids=[labels[0]],
            #                                                 output_logits=[output_logits[0]])

            # return (loss, outputs) if return_outputs else loss

    # trainer.compute_loss = types.MethodType(custom_compute_loss, trainer)

    trainer = CustomTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
    )

else:
    print("Setting up the trainer")

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
    )




# ---------------------------------------------------
# TRAINING
# ---------------------------------------------------

finetuned_model_dir = args.export_model_dir
log_dir = "../../model_ckpts/" + args.fine_tuned_model_ver + "/runs/"

torch.cuda.empty_cache()
print("Training starts")
trainer.train()
# trainer.train("../../model_ckpts/fine-tuning_wav2vec2_v17/checkpoint-15000")

log_history_fn = os.path.join(log_dir, "log_history.txt")
with open(log_history_fn, "w") as f:
    f.write(json.dumps(trainer.state.log_history))

print("Saving fine-tuned model")
model.save_pretrained(save_directory=finetuned_model_dir)
processor.save_pretrained(save_directory=finetuned_model_dir)

