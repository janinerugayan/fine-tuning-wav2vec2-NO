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
import torch.nn.functional as F
from scipy.spatial import distance
import ctc_optimized  # cython built ctc loss & grad calc
import ctc  # python-implemented ctc loss & grad calc
import asd_for_ctc  # to extract ASD metric aligned to label seq for CTC loss calc



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# INCORPORATING ASD COSDIST VALUES TO THE CTC CALCULATION
# and defining it as a custom autograd function
class MyCTC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, seq, input_length, cosdist_for_ctc, lambda_asd):

        # ============= CYTHON CTC loss implementation =============
        params = logits.transpose(1,0)
        # convert logits to log probs
        params = params - (torch.max(params, dim=0)[0])
        params = torch.exp(params)
        params = params / torch.sum(params, dim=0)

        params_arr = params.double().detach().cpu().numpy()
        seq_arr = seq.int().detach().cpu().numpy()
        cosdist_arr = np.array(cosdist_for_ctc, dtype=np.float64)
        # llForward, llBackward, alphas, betas = ctc_optimized.forward_pass(params_arr, seq_arr, blank=31)
        llForward, llBackward, alphas, betas = ctc_optimized.forward_pass_with_ASD(params_arr, seq_arr, cosdist_arr, lambda_asd, blank=31)

        alphas_tensor = torch.from_numpy(alphas).to(device)
        betas_tensor = torch.from_numpy(betas).to(device)
        llForward_tensor = torch.tensor(llForward).to(device)
        llBackward_tensor = torch.tensor(llBackward).to(device)

        ctx.save_for_backward(params, seq, input_length, alphas_tensor, betas_tensor, llForward_tensor, llBackward_tensor)

        return llForward_tensor

        # # ============= ctc loss implementation =============
        # llForward, llBackward, alphas, betas, params = ctc.forward_pass(logits, seq, device, blank=31)
        # ctx.save_for_backward(params, seq, input_length, alphas, betas, llForward, llBackward)

        # return llForward

    @staticmethod
    def backward(ctx, grad_output):

        # ============= CYTHON grad implementation =============
        params, seq, input_length, alphas, betas, llForward, llBackward = ctx.saved_tensors
        params_arr = params.double().detach().cpu().numpy()
        seq_arr = seq.int().detach().cpu().numpy()
        alphas_arr = alphas.double().detach().cpu().numpy()
        betas_arr = betas.double().detach().cpu().numpy()
        input_len_int = input_length.int().detach().cpu()

        grad = ctc_optimized.backward_pass(params_arr, seq_arr, alphas_arr, betas_arr, input_len_int, blank=31)

        grad_tensor = torch.tensor(grad).to(device)

        return (grad_tensor.transpose(1,0), None, None, None, None)

        # # ============= ctc grad implementation =============
        # params, seq, input_length, alphas, betas, llForward, llBackward = ctx.saved_tensors
        # grad = ctc.backward_pass(params, seq, alphas, betas, device, blank=31)

        # return (grad.transpose(1,0), None, None)


# USING STANF0RD-CTC CODE:
def compute_CTCloss_withASD(reference_text, predicted_text, ref_label_ids, output_logits, input_lengths, asd_model, asd_tokenizer, lambda_asd):  # originally includes: asd_model, asd_tokenizer
    loss = torch.zeros((len(reference_text)), requires_grad=True, device=device).double()
    for i in range(len(reference_text)):
        ref_text = reference_text[i].replace("[UNK]", "")
        pred_text = predicted_text[i].replace("[UNK]", "")
        label_ids = ref_label_ids[i]
        labels_mask = label_ids >= 0
        flattened_labels = label_ids.masked_select(labels_mask)
        logits = output_logits[i]
        ref_alignments = asd_for_ctc.get_asd_align(ref_text, pred_text, asd_model, asd_tokenizer)
        tokens_compressed = asd_for_ctc.get_per_token_cosdist(ref_alignments)
        cosdist_for_ctc = asd_for_ctc.get_cosdist_for_ctc(tokens_compressed, flattened_labels)
        myctcloss = MyCTC.apply
        loss[i] = myctcloss(logits, flattened_labels, input_lengths[i], cosdist_for_ctc, lambda_asd)
        # print("loss:", loss[i], "lambda:", lambda_asd)
    return loss.sum()


# ===================================================
# CTC with Gumbel-Softmax sampled ASD scoring
def compute_asd_score_batch(model, tokenizer, reference, hypothesis):
    asd_score = []
    for ref_text, hyp_text in zip(reference, hypothesis):
        ref_text = ref_text.replace("[UNK]", "")  # removes the [UNK] token in the reference text, observed during training
        hyp_text = hyp_text.replace("[UNK]", "")
        tokenized_ref = tokenizer(ref_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        tokenized_hyp = tokenizer(hyp_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
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
        asd_score.append((alignment.distance / num_tokens))
    return asd_score

def sampled_logits_asd_loss(reference_text, predicted_text, output_logits, metric_model, metric_tokenizer):
    # calculate asd scores for batch:
    asd_scores = compute_asd_score_batch(metric_model, metric_tokenizer, reference_text, predicted_text)
    # sample from output logits:
    sampled_logits = F.gumbel_softmax(output_logits, tau=1, hard=True, dim=-1)
    # sampled logits x ASD score:
    temp_list = []
    for i, logits in enumerate(sampled_logits):
        asd_matrix = torch.full_like(logits, (1 + (asd_scores[i]*10)), requires_grad=False)
        temp_list.append(logits.detach() * asd_matrix)
    sampled_logits_asd = torch.stack(temp_list, dim=0)
    sampled_logits_asd.unsqueeze(0)
    # calculate loss:
    # L1loss = torch.nn.SmoothL1Loss(reduction="mean", beta=1)
    L1loss = torch.nn.L1Loss(reduction="mean")
    loss = L1loss(input=sampled_logits, target=sampled_logits_asd) * 10
    # MSEloss = torch.nn.MSELoss(reduction="mean")
    # loss = MSEloss(input=sampled_logits, target=sampled_logits_asd) * 100
    return loss