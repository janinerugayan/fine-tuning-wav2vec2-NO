import transformers
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from datasets import load_dataset, load_metric, ClassLabel, Audio, Dataset
import pandas as pd
import math
import numpy as np
# from semdist import *
import librosa
import os
import re
import torch
import random
from pydub import AudioSegment
import xml.etree.ElementTree as ET
from test_utils import *





finetuned_model_dir = "../localhome/fine_tuned_models/wav2vec2_NO_v3/"
model_name = 'NbAiLab/nb-wav2vec2-1b-bokmaal'
train_dev_set = ["../../datasets/NordTrans_TUL/train/NRK/"]
log_file = "test_wav2vec2_v3.txt"


print("RUNNING MODELS WITH THE DEV DATA")

print("Loading Train/Dev Dataset")
dataset, full_dataset_df = load_train_eval_dataset(train_dev_set, test_size=0.1)
print(dataset)

print("Fine-tuned Model WER on Dev Set")

torch.cuda.empty_cache()
processor = Wav2Vec2Processor.from_pretrained(finetuned_model_dir)
model = Wav2Vec2ForCTC.from_pretrained(finetuned_model_dir)

wer_metric = load_metric("wer")
finetuned_results = dataset["test"].map(get_transcriptions_finetuned, remove_columns=dataset["test"].column_names)
print("dev set WER (fine-tuned): {:.3f}".format(
     wer_metric.compute(predictions=finetuned_results["asr_str"],
     references=finetuned_results["ref_str"])))
with open(log_file, "a") as f:
    f.write("dev set WER (fine-tuned): {:.3f}".format(
         wer_metric.compute(predictions=finetuned_results["asr_str"],
         references=finetuned_results["ref_str"])))

print("Original Model WER on Dev Set")

torch.cuda.empty_cache()
processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

wer_metric = load_metric("wer")
origmodel_results = dataset["test"].map(get_transcriptions_origmodel, remove_columns=dataset["test"].column_names)
print("dev set WER (original model): {:.3f}".format(
     wer_metric.compute(predictions=origmodel_results["asr_str"],
     references=origmodel_results["ref_str"])))
with open(log_file, "a") as f:
    f.write("dev set WER (original model): {:.3f}".format(
         wer_metric.compute(predictions=origmodel_results["asr_str"],
         references=origmodel_results["ref_str"])))



print("RUNNING MODELS WITH THE TEST DATA")

print("Loading test datasets")

rundkast_dir = ["../localhome/datasets/NordTrans_TUL/test/Rundkast/"]
dataset_rundkast = load_test_dataset(rundkast_dir)
dataset_rundkast = dataset_rundkast.map(remove_special_characters)

nbtale_dir = ["../localhome/datasets/NordTrans_TUL/test/NB_Tale/"]
dataset_nbtale = load_test_dataset(nbtale_dir)
dataset_nbtale = dataset_nbtale.map(remove_special_characters)

stortinget_dir = ["../localhome/datasets/NordTrans_TUL/test/Stortinget/"]
dataset_stortinget = load_test_dataset(stortinget_dir)
dataset_stortinget = dataset_stortinget.map(remove_special_characters)


print("Original model testing")
torch.cuda.empty_cache()
processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
wer_metric = load_metric("wer")

print("RUNDKAST")
Rundkast_results = dataset_rundkast.map(get_transcriptions_origmodel, remove_columns=dataset_rundkast.column_names)
print("Test WER (original): {:.3f}".format(
      wer_metric.compute(predictions=Rundkast_results["asr_str"],
      references=Rundkast_results["ref_str"])))
with open(log_file, "a") as f:
    f.write("Test WER (original): {:.3f}".format(
          wer_metric.compute(predictions=Rundkast_results["asr_str"],
          references=Rundkast_results["ref_str"])))

print("NB TALE")
NBTale_results = dataset_nbtale.map(get_transcriptions_origmodel, remove_columns=dataset_nbtale.column_names)
print("Test WER (original): {:.3f}".format(
     wer_metric.compute(predictions=NBTale_results["asr_str"],
     references=NBTale_results["ref_str"])))
with open(log_file, "a") as f:
    f.write("Test WER (original): {:.3f}".format(
         wer_metric.compute(predictions=NBTale_results["asr_str"],
         references=NBTale_results["ref_str"])))

print("STORTINGET")
Stortinget_results = dataset_stortinget.map(get_transcriptions_origmodel, remove_columns=dataset_stortinget.column_names)
print("Test WER (original): {:.3f}".format(
     wer_metric.compute(predictions=Stortinget_results["asr_str"],
     references=Stortinget_results["ref_str"])))
with open(log_file, "a") as f:
    f.write("Test WER (original): {:.3f}".format(
         wer_metric.compute(predictions=Stortinget_results["asr_str"],
         references=Stortinget_results["ref_str"])))


print("Fine-tuned model testing")
torch.cuda.empty_cache()
processor = Wav2Vec2Processor.from_pretrained(finetuned_model_dir)
model = Wav2Vec2ForCTC.from_pretrained(finetuned_model_dir)
wer_metric = load_metric("wer")

print("RUNDKAST")
finetuned_Rundkast_results = dataset_rundkast.map(get_transcriptions_finetuned, remove_columns=dataset_rundkast.column_names)
print("Test WER (fine-tuned): {:.3f}".format(
     wer_metric.compute(predictions=finetuned_Rundkast_results["asr_str"],
     references=finetuned_Rundkast_results["ref_str"])))
with open(log_file, "a") as f:
    f.write("Test WER (fine-tuned): {:.3f}".format(
         wer_metric.compute(predictions=finetuned_Rundkast_results["asr_str"],
         references=finetuned_Rundkast_results["ref_str"])))

print("NB TALE")
finetuned_NBTale_results = dataset_nbtale.map(get_transcriptions_finetuned, remove_columns=dataset_nbtale.column_names)
print("Test WER (fine-tuned): {:.3f}".format(
     wer_metric.compute(predictions=finetuned_NBTale_results["asr_str"],
     references=finetuned_NBTale_results["ref_str"])))
with open(log_file, "a") as f:
    f.write("Test WER (fine-tuned): {:.3f}".format(
         wer_metric.compute(predictions=finetuned_NBTale_results["asr_str"],
         references=finetuned_NBTale_results["ref_str"])))

print("STORTINGET")
finetuned_Stortinget_results = dataset_stortinget.map(get_transcriptions_finetuned, remove_columns=dataset_stortinget.column_names)
print("Test WER (fine-tuned): {:.3f}".format(
     wer_metric.compute(predictions=finetuned_Stortinget_results["asr_str"],
     references=finetuned_Stortinget_results["ref_str"])))
with open(log_file, "a") as f:
    f.write("Test WER (fine-tuned): {:.3f}".format(
         wer_metric.compute(predictions=finetuned_Stortinget_results["asr_str"],
         references=finetuned_Stortinget_results["ref_str"])))
