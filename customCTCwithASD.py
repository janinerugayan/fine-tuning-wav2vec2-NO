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
from tabulate import tabulate
from dtw import *
import torch
from scipy.spatial import distance


# returning ASD cosdist alignment to reference tokens
def get_asd_align(ref, hyp, asd_model, asd_tokenizer):
    tokenized_ref = asd_tokenizer(ref, padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokenized_hyp = asd_tokenizer(hyp, padding=True, truncation=True, max_length=512, return_tensors="pt")
    ref_input_ids = tokenized_ref["input_ids"].squeeze()

    with torch.no_grad():
        model_output_ref = asd_model(**tokenized_ref, output_hidden_states=True)
        model_output_hyp = asd_model(**tokenized_hyp, output_hidden_states=True)
    hidden_states_ref = model_output_ref.hidden_states
    hidden_states_hyp = model_output_hyp.hidden_states
    all_layers_reference = []
    all_layers_hypothesis = []
    for i in range(1,13):
        all_layers_reference.append(hidden_states_ref[i].squeeze())
        all_layers_hypothesis.append(hidden_states_hyp[i].squeeze())
    ref_embedding_sequence = torch.stack(all_layers_reference).mean(dim=0)
    hyp_embedding_sequence = torch.stack(all_layers_hypothesis).mean(dim=0)

    alignment = dtw(hyp_embedding_sequence, ref_embedding_sequence, dist_method=distance.cosine, keep_internals=True)
    num_tokens = len(ref_embedding_sequence)

    ref_alignment_idxs = alignment.index2
    hyp_alignment_idxs = alignment.index1

    ref_alignments = []
    for i in range(len(ref_alignment_idxs)):
        ref_embedding = ref_embedding_sequence[ref_alignment_idxs[i]]
        hyp_embedding = hyp_embedding_sequence[hyp_alignment_idxs[i]]
        cosdist = distance.cosine(ref_embedding, hyp_embedding)
        ref_token = asd_tokenizer.convert_ids_to_tokens(ref_input_ids[ref_alignment_idxs[i]].reshape(1))[0]
        ref_alignments.append((ref_alignment_idxs[i], ref_token, cosdist))

    return ref_alignments


def get_per_token_cosdist(asd_alignments):
    # collapse repetitions in tokens and wordpieces in the HYP alignment from ASD
    clean_alignment = []
    for i, item in enumerate(asd_alignments):
        if i < (len(asd_alignments) - 1):
            if len(clean_alignment) == 0:
                if item[1] != "[CLS]" and item[1] != "[SEP]":
                    clean_alignment.append(item)
            else:
                if item[0] == clean_alignment[-1][0]:
                    averaged_cosdist = sum([item[2], clean_alignment[-1][2]]) / 2
                    clean_alignment.pop(-1)
                    clean_alignment.append((item[0], item[1], averaged_cosdist))
                else:
                    clean_alignment.append(item)

    # GROUPING THE TOKENS FROM ASD CALCULATION, SUCH THAT WORDPIECES ARE TOGETHER
    regrouped_tokens = []
    for i, item in enumerate(clean_alignment):
        if item[1] != "[CLS]" and item[1] != "[SEP]":
            if "##" not in item[1]:
                if i < (len(clean_alignment)-1) and "##" in clean_alignment[i+1][1]:  # start of a group of wordpieces
                    wordpiece_group = []
                    wordpiece_group.append(item)
                    regrouped_tokens.append(wordpiece_group)
                else:
                    regrouped_tokens.append(item)
            elif "##" in item[1]:  # parts of the word
                wordpiece_group.append(item)
    # for i, item in enumerate(clean_alignment):
    #     if item[1] != "[CLS]" and item[1] != "[SEP]":
    #         if "##" not in item[1] and "##" in clean_alignment[i+1][1]:  # start of a group of wordpieces
    #             wordpiece_group = []
    #             wordpiece_group.append(item)
    #             regrouped_tokens.append(wordpiece_group)
    #         elif "##" in item[1]:  # parts of the word
    #             wordpiece_group.append(item)
    #         else:  # not wordpieces
    #             regrouped_tokens.append(item)

    # COLLAPSE WORDPIECES INTO WORDS & TAKE AVERAGE OF COSDIST
    tokens_compressed = []
    for token_group in regrouped_tokens:
        if isinstance(token_group, list):
            wp_combined = ''.join([wordpiece[1].replace("##", "") for wordpiece in token_group])
            token_ave_cosdist = sum([wordpiece[2] for wordpiece in token_group]) / len(token_group)
            tokens_compressed.append(("combined", wp_combined, token_ave_cosdist))
        else:
            tokens_compressed.append(token_group)

    return tokens_compressed


# aligning ASD cosdist values to label sequence
def get_cosdist_for_ctc(tokens_compressed, label_ids):
    cosdist_for_ctc = []
    token_count = 0
    for label in label_ids:
        if label == 0:
            token_count += 1
            cosdist_for_ctc.append(0)
        else:
            cosdist_for_ctc.append(tokens_compressed[token_count][2])
    if len(cosdist_for_ctc) != len(label_ids):
        print("mismatch in number of tokens compressed and tokens identified from label ids")
        return cosdist_for_ctc
    else:
        return cosdist_for_ctc


# INCORPORATING ASD COSDIST VALUES TO THE CTC CALCULATION
def ctc_loss_with_ASD(params, seq, cosdist_for_ctc, blank=0):
    seqLen = seq.shape[0]  # length of label sequence
    L = 2*seqLen + 1  # length of the label sequence with blanks
    T = params.shape[1]  # length of utterance (time)

    alphas = torch.zeros((L,T))
    betas = torch.zeros((L,T))

    # convert logits to log probs
    params = params - (torch.max(params, dim=0)[0])
    params = torch.exp(params)
    params = params / torch.sum(params, dim=0)

    # initialize alphas and forward pass
    alphas[0,0] = params[blank,0]
    alphas[1,0] = params[seq[0],0]
    c = torch.sum(alphas[:,0])
    alphas[:,0] = alphas[:,0].clone() / c
    llForward = torch.log(c)

    for t in range(1,T):
        start = max(0,L-2*(T-t))
        end = min(2*t+2,L)
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t] = alphas[s,t-1].clone() * params[blank,t]
                else:
                    alphas[s,t] = (alphas[s,t-1].clone() + alphas[s-1,t-1].clone()) * params[blank,t]
            # same label twice
            elif s == 1 or seq[l] == seq[l-1]:
                alphas[s,t] = (alphas[s,t-1].clone() + alphas[s-1,t-1].clone()) * params[seq[l],t] * (1 - cosdist_for_ctc[l])  # scale 0 to 1
            else:
                alphas[s,t] = (alphas[s,t-1].clone() + alphas[s-1,t-1].clone() + alphas[s-2,t-1].clone()) * params[seq[l],t] * (1 - cosdist_for_ctc[l])

        # normalize at current time (prevent underflow)
        c = torch.sum(alphas[start:end,t])
        alphas[start:end,t] = alphas[start:end,t].clone() / c
        llForward = llForward + torch.log(c)

    # initialize betas and backwards pass
    betas[-1,-1] = params[blank,-1]
    betas[-2,-1] = params[seq[-1],-1]
    c = torch.sum(betas[:,-1])
    betas[:,-1] = betas[:,-1].clone() / c
    llBackward = torch.log(c)

    for t in range(T-2,-1,-1):
        start = max(0,L-2*(T-t))
        end = min(2*t+2,L)
        for s in range(end-1,-1,-1):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s == L-1:
                    betas[s,t] = betas[s,t+1].clone() * params[blank,t]
                else:
                    betas[s,t] = (betas[s,t+1].clone() + betas[s+1,t+1].clone()) * params[blank,t]
            # same label twice
            elif s == L-2 or seq[l] == seq[l+1]:
                betas[s,t] = (betas[s,t+1].clone() + betas[s+1,t+1].clone()) * params[seq[l],t] * (1 - cosdist_for_ctc[l])
            else:
                betas[s,t] = (betas[s,t+1].clone() + betas[s+1,t+1].clone() + betas[s+2,t+1].clone()) * params[seq[l],t] * (1 - cosdist_for_ctc[l])

        # normalize at current time
        c = torch.sum(betas[start:end,t])
        betas[start:end,t] = betas[start:end,t].clone() / c
        llBackward = llBackward + torch.log(c)

    # Compute gradient with respect to unnormalized input parameters
    grad = torch.zeros(params.shape)
    ab = alphas*betas
    for s in range(L):
        l = int((s-1)/2)
        # blank
        if s%2 == 0:
            grad[blank,:] = grad[blank,:] + ab[s,:]
            ab[s,:] = ab[s,:]/params[blank,:]
        else:
            grad[seq[l],:] = grad[seq[l],:] + ab[s,:]
            ab[s,:] = ab[s,:]/(params[seq[l],:])
    absum = torch.sum(ab,axis=0)

    # Check for underflow or zeros in denominator of gradient
    llDiff = torch.abs(llForward-llBackward)
    if llDiff > 1e-5 or torch.sum(absum==0) > 0:
        print("Diff in forward/backward LL : %f"%llDiff)
        print("Zeros found : (%d/%d)"%(torch.sum(absum==0),absum.shape[0]))
        # return (-llForward, grad)
        return -llForward
    else:
        grad = params - grad / (params * absum)
        # return (-llForward, grad)
        return -llForward


def compute_CTCloss_withASD(reference_text, predicted_text, ref_label_ids, output_logits, asd_model, asd_tokenizer):
    loss = 0
    for i in range(len(reference_text)):
        ref_text = reference_text[i].replace("[UNK]", "")
        pred_text = predicted_text[i].replace("[UNK]", "")
        label_ids = ref_label_ids[i]
        logits = output_logits[i]
        print(ref_text)
        print(pred_text)
        ref_alignments = get_asd_align(ref_text, pred_text, asd_model, asd_tokenizer)
        tokens_compressed = get_per_token_cosdist(ref_alignments)
        cosdist_for_ctc = get_cosdist_for_ctc(tokens_compressed, label_ids)
        loss = loss + ctc_loss_with_ASD(logits.transpose(1,0), label_ids, cosdist_for_ctc, blank=0)
    loss = loss / len(reference_text)
    return loss