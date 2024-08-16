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
# from pydub import AudioSegment
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

from torchaudio.models.decoder import ctc_decoder



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_asd_score_single_utt(model, tokenizer, reference, hypothesis):
    ref_text = reference.replace("[UNK]", "")
    hyp_text = hypothesis.replace("[UNK]", "")
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
    asd_score = alignment.distance / num_tokens
    return asd_score


# INCORPORATING ASD COSDIST VALUES TO THE CTC CALCULATION
# and defining it as a custom autograd function
class MyCTC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, seq, input_length, cosdist_for_ctc):

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
        llForward, llBackward, alphas, betas = ctc_optimized.forward_pass_with_ASD(params_arr, seq_arr, cosdist_arr, blank=31)

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
def compute_CTCloss_withASD(reference_text, predicted_text, ref_label_ids, output_logits, input_lengths, asd_model, asd_tokenizer):  # originally includes: asd_model, asd_tokenizer
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
        if len(flattened_labels) != len(cosdist_for_ctc):
            raise Exception("cosdist for ctc length not equal to flattened labels length")
        loss[i] = myctcloss(logits, flattened_labels, input_lengths[i], cosdist_for_ctc)
        # loss[i] = myctcloss(logits, flattened_labels, input_lengths[i])
        # print("loss:", loss[i], "lambda:", lambda_asd)
        # print("ref:", ref_text)
        # print("hyp:", pred_text)
        # print("loss:", loss[i])
    return loss.sum()


def compute_CTCloss_nbest(reference_text, output_logits, input_lengths, asd_model, asd_tokenizer):
    decoder = ctc_decoder(lexicon=None, tokens="tokens.txt", nbest=10, beam_size=100, blank_token="[PAD]",
                          sil_token="|", unk_word="[UNK]")
    targets = []
    target_lengths = []
    log_probs = F.log_softmax(output_logits, dim=-1, dtype=torch.float32).transpose(0, 1)

    for i in range(len(reference_text)):
        ref_text = reference_text[i].replace("[UNK]", "")
        logits = output_logits[i]
        # get nbest hypotheses and rank them
        nbest_list = decoder(logits.type(torch.float32).detach().cpu()[None, :, :])
        nbest_token_list = []
        asd_score_list = [0] * len(nbest_list[0])
        hyp_list = []
        for j, item in enumerate(nbest_list[0]):
            tokens = item.tokens
            for k in range(len(tokens)):
                if tokens[k] == 0:
                    tokens_mod = tokens[k+1:]
                else:
                    break
            chars = decoder.idxs_to_tokens(tokens_mod)
            nbest_token_list.append(tokens_mod)
            hyp_text = re.sub(" +", " ", "".join(chars).replace("|", " "))
            hyp_list.append(hyp_text)
            asd_score_list[j] = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text, hyp_text)
        targets.append(torch.tensor(nbest_token_list[np.argmin(asd_score_list)]))
        target_lengths.append(torch.tensor(len(nbest_token_list[np.argmin(asd_score_list)])))

    targets_tensor = torch.cat(targets, dim=0)
    targets_len_tensor = torch.tensor(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        loss = F.ctc_loss(
                log_probs,
                targets_tensor,
                input_lengths,
                targets_len_tensor,
                blank=31,
                reduction="mean",
                zero_infinity=True,
                )

    return loss


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


def compute_sampled_meanASD(reference_text, output_logits, asd_model, asd_tokenizer, processor):
    num_samples = 5
    loss = torch.zeros((len(reference_text)), requires_grad=True, device=device).double()
    for i in range(len(reference_text)):
        ref_text = reference_text[i].replace("[UNK]", "")
        logits = output_logits[i]
        asd_scores = [0] * num_samples
        # get samples using gumble softmax sampling
        for j in range(num_samples):
            sampled_logits = F.gumbel_softmax(logits, tau=10, hard=True, dim=-1)
            hyp_text = processor.decode(sampled_logits.detach().cpu().numpy()).text
            asd_scores[j] = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text, hyp_text)
        loss[i] = np.mean(asd_scores)
    return torch.mean(loss)


def get_mass_prob(paths, softmax_ctc, model_pred_length, eps: float = 1e-7):
    """
    compute the path probability mass
    :param paths: ctc alignments
    :param softmax_ctc: model logits after softmax
    :param model_pred_length:  max length of all given paths
    :return: avg of the paths probability
    """
    log_indexes_probs = softmax_ctc.gather(dim=1, index=paths) + eps
    if len(log_indexes_probs) > model_pred_length:
        log_indexes_probs[model_pred_length:, :] = torch.zeros((log_indexes_probs.shape[0] - model_pred_length, 1))
    return torch.sum(log_indexes_probs, dim=0) / (model_pred_length.unsqueeze(-1))


def sampled_pair_hinge_loss(ref_text, output_logits, input_lengths, asd_model, asd_tokenizer, processor):
    loss = torch.zeros((len(ref_text)), requires_grad=True, device=device).double()
    for i in range(len(ref_text)):
        log_probs = F.log_softmax(output_logits[i], dim=-1, dtype=torch.float32)
        non_zero_logits = []
        non_zero_asd = []
        while len(non_zero_asd) < 2:
            sampled_logits = F.gumbel_softmax(output_logits[i], tau=100, hard=True, dim=-1)
            hyp_text = processor.decode(sampled_logits.detach().cpu().numpy()).text
            asd_score = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text[i], hyp_text)
            if asd_score != 0:
                non_zero_logits.append(sampled_logits)
                non_zero_asd.append(asd_score)
        mass_prob_0 = get_mass_prob(non_zero_logits[0].type(torch.LongTensor).to(device), log_probs, input_lengths[i])
        mass_prob_1 = get_mass_prob(non_zero_logits[1].type(torch.LongTensor).to(device), log_probs, input_lengths[i])
        if non_zero_asd[0] < non_zero_asd[1]:
            subtract_path_probs = mass_prob_1 - mass_prob_0
        else:
            subtract_path_probs = mass_prob_0 - mass_prob_1
        loss[i] = torch.sum(torch.clamp(subtract_path_probs, min=0))
    return torch.mean(loss)


def sampled_multi_hinge_loss(ref_text, output_logits, input_lengths, asd_model, asd_tokenizer, processor):
    num_samples = 10
    loss = torch.zeros((len(ref_text)), requires_grad=True, device=device).double()
    for i in range(len(ref_text)):
        log_probs = F.log_softmax(output_logits[i], dim=-1, dtype=torch.float32)
        non_zero_logits = []
        non_zero_asd = []
        while len(non_zero_asd) < num_samples:
            sampled_logits = F.gumbel_softmax(output_logits[i], tau=100, hard=True, dim=-1)
            hyp_text = processor.decode(sampled_logits.detach().cpu().numpy()).text
            asd_score = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text[i], hyp_text)
            if asd_score != 0:
                non_zero_logits.append(sampled_logits)
                non_zero_asd.append(asd_score)
        lowest_asd_idx = np.argmin(np.array(non_zero_asd))
        mass_prob_list = []
        for j in range(len(non_zero_asd)):
            mass_prob_list.append(get_mass_prob(non_zero_logits[j].type(torch.LongTensor).to(device), log_probs, input_lengths[i]))
        local_loss = torch.zeros((len(mass_prob_list)-1), requires_grad=True, device=device).double()
        l = 0
        for k in range(len(mass_prob_list)):
            if k != lowest_asd_idx:
                subtract_path_probs = mass_prob_list[k] - mass_prob_list[lowest_asd_idx]
                local_loss[l] = torch.sum(torch.clamp((subtract_path_probs), min=0))
                l += 1
        loss[i] = torch.mean(local_loss)
    return torch.mean(loss)


# get mass probability of sampled path & calculate the expected ASD score
def sampled_multi_expected_ASD(ref_text, output_logits, input_lengths, asd_model, asd_tokenizer, processor):
    num_samples = 10
    loss = torch.zeros((len(ref_text)), requires_grad=True, device=device).double()
    for i in range(len(ref_text)):
        log_probs = F.log_softmax(output_logits[i], dim=-1, dtype=torch.float32)
        samples_loss = torch.zeros(num_samples, requires_grad=True, device=device).double()
        for j in range(num_samples):
            sampled_logits = F.gumbel_softmax(output_logits[i], tau=10, hard=True, dim=-1)
            hyp_text = processor.decode(sampled_logits.detach().cpu().numpy()).text
            asd_score = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text[i], hyp_text)
            mass_prob = get_mass_prob(sampled_logits.type(torch.LongTensor).to(device), log_probs, input_lengths[i])
            mass_prob_sum = torch.sum(mass_prob)
            path_prob = mass_prob_sum * sampled_logits
            path_prob_norm = torch.exp(path_prob / torch.sum(path_prob)) * asd_score
            samples_loss[j] = torch.mean(path_prob_norm)
            # samples_loss[j] = torch.sum(torch.clamp(mass_prob, min=0)) * asd_score
        loss[i] = torch.mean(samples_loss)
    return torch.mean(loss)


def sampled_multi_expected_ASD_ver2(ref_text, output_logits, input_lengths, asd_model, asd_tokenizer, processor):
    num_samples = 5
    loss = torch.zeros((len(ref_text)), requires_grad=True, device=device).double()
    for i in range(len(ref_text)):
        log_probs = F.log_softmax(output_logits[i], dim=-1, dtype=torch.float32)
        samples_loss = torch.zeros(num_samples, requires_grad=True, device=device).double()
        sample_probs = torch.zeros(num_samples, requires_grad=True, device=device).double()
        asd_scores = []
        for j in range(num_samples):
            sampled_logits = F.gumbel_softmax(output_logits[i], tau=10, hard=True, dim=-1)
            hyp_text = processor.decode(sampled_logits.detach().cpu().numpy()).text
            asd_scores.append(compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text[i], hyp_text))
            mass_prob = get_mass_prob(sampled_logits.type(torch.LongTensor).to(device), log_probs, input_lengths[i])
            sample_probs[j] = torch.sum(mass_prob)
        for k in range(num_samples):
            samples_loss[k] = torch.exp((sample_probs[k] / torch.sum(sample_probs)) * asd_scores[k])
        loss[i] = torch.mean(samples_loss)
    return torch.mean(loss)


# ===================================================
# Minimum ASD Loss implementation adapted from k2 MWER Loss

