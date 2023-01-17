import transformers
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from transformers import AutoTokenizer, BertModel
from datasets import load_dataset, load_metric, ClassLabel, Audio, Dataset
import pandas as pd
import math
import numpy as np
import librosa
import os
import re
import torch
import random
from pydub import AudioSegment
import xml.etree.ElementTree as ET
import argparse



def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

# for dataset used for testing after fine-tuning the model
def load_test_dataset(data_dir_list: list[str]):
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
                elif fn.endswith(".txt"):
                    text_id = os.path.splitext(fn)[0]
                    with open(os.path.join(root, fn), encoding="utf-8") as text_file:
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
    return dataset

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

def xml_to_dataframe(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []
    for pa in root.findall('./ch/se/pa'):
        for p in pa.findall('./p'):
            if 'M' in p.attrib['b']:
                start = p.attrib['b'][2:-1].split('M')
                if len(start) == 2:
                    minutes = float(start[0]) * 60
                    start_secs = minutes + float(start[1])
                else:
                    start_secs = float(start[0]) * 60
            else:
                start_secs = float(p.attrib['b'][2:-1])
            if 'M' in p.attrib['e']:
                end = p.attrib['e'][2:-1].split('M')
                if len(end) == 2:
                    minutes = float(end[0]) * 60
                    end_secs = minutes + float(end[1])
                else:
                    end_secs = float(end[0]) * 60
            else:
                end_secs = float(p.attrib['e'][2:-1])
            text = p.text
            data.append([text, start_secs, end_secs])
    df = pd.DataFrame(data, columns = ['word', 'startTime', 'endTime'])
    return df

# for dataset used in fine-tuning the model
def load_dev_set(dev_set_path:str):
    dev_set_df = pd.read_csv(dev_set_path)
    dev_set_df.set_index(dev_set_df["segment_id"], drop=True, inplace=True)
    dev_set_df.drop(labels=["Unnamed: 0", "segment_id"], axis="columns", inplace=True)
    dataset = Dataset.from_pandas(dev_set_df)
    return dataset

# for dividing the test dataset into meaningful segments
def get_wav_text_segments(timebounds_dir:str, segmentbounds_dir:str, source_wav_dir:str, export_dir:str):
    for (root, dirs, files) in os.walk(segmentbounds_dir, topdown=True):
        for file in files:
            if file.endswith("csv"):

                fn = os.path.splitext(file)[0]
                print(f"Processing: {fn}")

                base_df = pd.read_csv(os.path.join(segmentbounds_dir, file))
                base_df.replace(np.nan, " ", regex=True, inplace=True)
                for row in base_df.itertuples():
                    if row[1] == " ":
                        base_df.drop(row.Index, axis=0, inplace=True)
                base_df.reset_index(inplace=True)
                xml_file = fn + ".trsx"
                timebounds_df = xml_to_dataframe(os.path.join(timebounds_dir, xml_file))
                timebounds_df.columns = ["ref_text", "startTime", "endTime"]
                dataset_df = timebounds_df.merge(base_df["sentence_boundary"], left_index=True, right_index=True)

                start_time = []
                end_time = []
                start_index = []
                end_index = []
                for row in dataset_df.itertuples():
                    if row[4] == "b":
                        start_time.append(row[2])
                        start_index.append(row.Index)
                    elif row[4] == "e":
                        end_time.append(row[3])
                        end_index.append(row.Index + 1)

                if len(start_time) > len(end_time):
                    end_time.append(dataset_df.iloc[-1]["endTime"])

                if len(start_index) > len(end_index):
                    end_index.append(dataset_df.index[-1] + 1)

                if len(start_time) != len(start_index):
                    print("Number of timestamps and index pairs don't match!")
                    exit()

                AUDIO_FILE = source_wav_dir + fn + ".wav"
                sound = AudioSegment.from_file(AUDIO_FILE)
                transcriptions = []
                for i in range(len(start_time)):
                    start = start_time[i] * 1000
                    end = end_time[i] * 1000
                    cut = sound[start:end]
                    cut.export(export_dir + fn + "_cut_" + str(i) + ".wav", format="wav")
                    text = " ".join(dataset_df.iloc[start_index[i]:end_index[i]]["ref_text"].tolist())
                    text_strip = re.sub(" +", " ", text)
                    transcriptions.append(text_strip)
                for i in range(len(transcriptions)):
                    text_file = export_dir + fn + "_cut_" + str(i) + ".txt"
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write(transcriptions[i])

# for dividing the test dataset into just chunks of 20 words
def group_by_20(timebounds_dir, source_wav_dir, export_dir):
    for (root, dirs, files) in os.walk(timebounds_dir, topdown=True):
        for file in files:
            print(f"Processing: {file}")
            if file.endswith("checkpoint.trsx"):
                continue
            elif file.endswith(".trsx"):
                xml_file = os.path.join(timebounds_dir, file)
                df = xml_to_dataframe(xml_file)

                timestamps = []
                transcriptions = []
                for i, g in df.groupby(np.arange(len(df)) // 20):
                    if len(g) < 20:
                        start = g.iloc[0]["startTime"]
                        end = g.iloc[len(g)-1]["endTime"]
                        text = " ".join(g.iloc[0:len(g)]["word"].tolist())
                        text_strip = re.sub(" +", " ", text)
                    else:
                        start = g.iloc[0]["startTime"]
                        end = g.iloc[19]["endTime"]
                        text = " ".join(g.iloc[0:20]["word"].tolist())
                        text_strip = re.sub(" +", " ", text)
                    timestamps.append((start, end))
                    transcriptions.append(text_strip)

                fn = os.path.splitext(file)[0]

                for i in range(len(transcriptions)):
                    text_file = export_dir + fn + "_cut_" + str(i) + ".txt"
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write(transcriptions[i])

                AUDIO_FILE = source_wav_dir + fn + ".wav"
                sound = AudioSegment.from_file(AUDIO_FILE)
                for i in range(len(timestamps)):
                    start = timestamps[i][0] * 1000
                    end = timestamps[i][1] * 1000
                    cut = sound[start:end]
                    cut.export(export_dir + fn + "_cut_" + str(i) + ".wav", format="wav")

def get_transcriptions(batch):
    audiofile = batch["path"]
    reference_text = batch["text"]
    audio, rate = librosa.load(audiofile, sr=16000)
    input_values = processor(audio, sampling_rate=rate, return_tensors='pt').input_values
    with torch.no_grad():
        logits = model(input_values).logits
    transcription = processor.batch_decode(logits.detach().numpy()).text
    batch["asr_str"] = transcription[0]
    batch["ref_str"] = reference_text
    batch["asd"] = asd_metric.compute(model=metric_model, tokenizer=metric_tokenizer, reference=reference_text, hypothesis=transcription[0])
    return batch

def get_score_per_utt(batch):
    # batch["wer"] = wer_metric.compute(predictions=batch["asr_str"], references=batch["ref_str"])
    batch["asd"] = asd_metric.compute(model=metric_model, tokenizer=metric_tokenizer, reference=batch["ref_str"], hypothesis=batch["asr_str"])
    return batch





parser = argparse.ArgumentParser()
parser.add_argument("--original_model",     type=str)
parser.add_argument("--fine_tuned_model",   type=str)
parser.add_argument("--log_file",           type=str)
parser.add_argument("--get_orig_model_results", type=int)
# parser.add_argument("--metric_to_use",             type=str)
# parser.add_argument("--extract_transcriptions",    type=int)
args = parser.parse_args()

model_name = args.original_model
finetuned_model_dir = args.fine_tuned_model
log_file = args.log_file
# metric_to_use = args.metric_to_use
# extract_transcriptions = args.extract_transcriptions

rundkast_dir = ["../../datasets/NordTrans_TUL/test/Rundkast/"]
nbtale_dir = ["../../datasets/NordTrans_TUL/test/NB_Tale/"]
stortinget_dir = ["../../datasets/NordTrans_TUL/test/Stortinget/"]

original_model_name = os.path.basename(args.original_model)
finetuned_model_name = os.path.basename(args.fine_tuned_model)




print("RUNNING MODELS WITH THE TEST DATA")

print("Loading test datasets")

dataset_rundkast = load_test_dataset(rundkast_dir)
dataset_rundkast = dataset_rundkast.map(remove_special_characters)

dataset_nbtale = load_test_dataset(nbtale_dir)
dataset_nbtale = dataset_nbtale.map(remove_special_characters)

dataset_stortinget = load_test_dataset(stortinget_dir)
dataset_stortinget = dataset_stortinget.map(remove_special_characters)

print("Loading evaluation metrics")

# LOAD WER METRIC
wer_metric = load_metric("wer")

# LOAD ASD METRIC
transformers.logging.set_verbosity(40)
metric_modelname = 'ltgoslo/norbert'
metric_model = BertModel.from_pretrained(metric_modelname)
metric_tokenizer = AutoTokenizer.from_pretrained(metric_modelname)
asd_metric = load_metric("asd_metric.py")




if args.get_orig_model_results == 1:

    print("Original model testing")
    torch.cuda.empty_cache()
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    print("RUNDKAST")
    Rundkast_results = dataset_rundkast.map(get_transcriptions)
    # Rundkast_results = Rundkast_results.map(get_score_per_utt)
    Rundkast_results.to_csv("./logs/Rundkast_results_" + original_model_name + ".csv" )
    # wer_score = Rundkast_results["wer"].mean()
    asd_score = Rundkast_results["asd"].mean()
    # print("Test Score (original) WER: {:.3f}".format(wer_score))
    print("Test Score (original) ASD: {:.3f}".format(asd_score))
    with open(log_file, "a") as f:
        # f.write("Rundkast Test Score (original) WER: {:.3f}\n".format(wer_score))
        f.write("Rundkast Test Score (original) ASD: {:.3f}\n".format(asd_score))

#     print("NB TALE")
#     NBTale_results = dataset_nbtale.map(get_transcriptions)
#     NBTale_results = NBTale_results.map(get_score_per_utt)
#     NBTale_results.to_csv("./logs/NBTale_results_" + original_model_name + ".csv" )
#     wer_score = NBTale_results["wer"].mean()
#     asd_score = NBTale_results["asd"].mean()
#     print("Test Score (original) WER: {:.3f}".format(wer_score))
#     print("Test Score (original) ASD: {:.3f}".format(asd_score))
#     with open(log_file, "a") as f:
#         f.write("NB Tale Test Score (original) WER: {:.3f}\n".format(wer_score))
#         f.write("NB Tale Test Score (original) ASD: {:.3f}\n".format(asd_score))

#     print("STORTINGET")
#     Stortinget_results = dataset_stortinget.map(get_transcriptions)
#     Stortinget_results = Stortinget_results.map(get_score_per_utt)
#     Stortinget_results.to_csv("./logs/Stortinget_results_" + original_model_name + ".csv" )
#     wer_score = Stortinget_results["wer"].mean()
#     asd_score = Stortinget_results["asd"].mean()
#     print("Test Score (original) WER: {:.3f}".format(wer_score))
#     print("Test Score (original) ASD: {:.3f}".format(asd_score))
#     with open(log_file, "a") as f:
#         f.write("Stortinget Test Score (original) WER: {:.3f}\n".format(wer_score))
#         f.write("Stortinget Test Score (original) ASD: {:.3f}\n".format(asd_score))




# print("Fine-tuned model testing")
# torch.cuda.empty_cache()
# processor = Wav2Vec2ProcessorWithLM.from_pretrained(finetuned_model_dir)
# model = Wav2Vec2ForCTC.from_pretrained(finetuned_model_dir)

# print("RUNDKAST")
# finetuned_Rundkast_results = dataset_rundkast.map(get_transcriptions)
# finetuned_Rundkast_results = finetuned_Rundkast_results.map(get_score_per_utt)
# finetuned_Rundkast_results.to_csv("./logs/finetuned_Rundkast_results_" + finetuned_model_name + ".csv" )
# wer_score = finetuned_Rundkast_results["wer"].mean()
# asd_score = finetuned_Rundkast_results["asd"].mean()
# print("Test Score (fine-tuned) WER: {:.3f}".format(wer_score))
# print("Test Score (fine-tuned) ASD: {:.3f}".format(asd_score))
# with open(log_file, "a") as f:
#     f.write("Rundkast Test Score (fine-tuned) WER: {:.3f}\n".format(wer_score))
#     f.write("Rundkast Test Score (fine-tuned) ASD: {:.3f}\n".format(asd_score))

# print("NB TALE")
# finetuned_NBTale_results = dataset_nbtale.map(get_transcriptions)
# finetuned_NBTale_results = finetuned_NBTale_results.map(get_score_per_utt)
# finetuned_NBTale_results.to_csv("./logs/finetuned_NBTale_results_" + finetuned_model_name + ".csv" )
# wer_score = finetuned_NBTale_results["wer"].mean()
# asd_score = finetuned_NBTale_results["asd"].mean()
# print("Test Score (fine-tuned) WER: {:.3f}".format(wer_score))
# print("Test Score (fine-tuned) ASD: {:.3f}".format(asd_score))
# with open(log_file, "a") as f:
#     f.write("NB Tale Test Score (fine-tuned) WER: {:.3f}\n".format(wer_score))
#     f.write("NB Tale Test Score (fine-tuned) ASD: {:.3f}\n".format(asd_score))

# print("STORTINGET")
# finetuned_Stortinget_results = dataset_stortinget.map(get_transcriptions)
# finetuned_Stortinget_results = finetuned_Stortinget_results.map(get_score_per_utt)
# finetuned_Stortinget_results.to_csv("./logs/finetuned_Stortinget_results_" + finetuned_model_name + ".csv" )
# wer_score = finetuned_Stortinget_results["wer"].mean()
# asd_score = finetuned_Stortinget_results["asd"].mean()
# print("Test Score (fine-tuned) WER: {:.3f}".format(wer_score))
# print("Test Score (fine-tuned) ASD: {:.3f}".format(asd_score))
# with open(log_file, "a") as f:
#     f.write("Stortinget Test Score (fine-tuned) WER: {:.3f}\n".format(wer_score))
#     f.write("Stortinget Test Score (fine-tuned) ASD: {:.3f}\n".format(asd_score))

