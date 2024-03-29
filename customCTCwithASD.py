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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    for i, label in enumerate(label_ids):
        # for the first utterance
        if len(cosdist_for_ctc) == 0 or all(cosdist == 0 for cosdist in cosdist_for_ctc):
            if label == 0 or label > 29:
                cosdist_for_ctc.append(0)
            else:
                cosdist_for_ctc.append(tokens_compressed[token_count][2])
        # for the next utterances
        else:
            if label == 0:
                cosdist_for_ctc.append(0)
                if i < (len(label_ids)-1) and 0 < label_ids[i+1] < 30:
                    token_count += 1
            else:
                cosdist_for_ctc.append(tokens_compressed[token_count][2])
    return cosdist_for_ctc


# INCORPORATING ASD COSDIST VALUES TO THE CTC CALCULATION
# and defining it as a custom autograd function
class MyCTC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, seq, blank=31):
        params = logits.transpose(1,0)
        seqLen = seq.shape[0]  # length of label sequence
        # seqLen = len(seq)
        # print(seq.shape[0], seqLen)
        L = 2*seqLen + 1  # length of the label sequence with blanks
        T = params.shape[1]  # length of utterance (time)

        alphas = torch.zeros((L,T), device=device).double()
        betas = torch.zeros((L,T), device=device).double()

        # convert logits to log probs
        params = params - (torch.max(params, dim=0)[0])
        params = torch.exp(params)
        params = params / torch.sum(params, dim=0)
        # log probs calculated here not the same with log probs input to pytorch CTC function
        # but they both produce the same CTC loss

        # initialize alphas and forward pass
        alphas[0,0] = params[blank,0]
        alphas[1,0] = params[seq[0],0]
        c = torch.sum(alphas[:,0])
        alphas[:,0] = alphas[:,0] / c
        llForward = torch.log(c)

        for t in range(1,T):
            start = max(0,L-2*(T-t))
            end = min(2*t+2,L)
            for s in range(start,L):
                l = int((s-1)/2)
                # blank
                if s%2 == 0:
                    if s==0:
                        alphas[s,t] = alphas[s,t-1] * params[blank,t]
                    else:
                        alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t]
                # same label twice
                elif s == 1 or seq[l] == seq[l-1]:
                    # alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t] * (1 + cosdist_for_ctc[l])
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
                else:
                    # alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) * params[seq[l],t] * (1 + cosdist_for_ctc[l])
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) * params[seq[l],t]

            # normalize at current time (prevent underflow)
            c = torch.sum(alphas[start:end,t])
            alphas[start:end,t] = alphas[start:end,t] / c
            llForward = llForward + torch.log(c)

        # initialize betas and backwards pass
        betas[-1,-1] = params[blank,-1]
        betas[-2,-1] = params[seq[-1],-1]
        c = torch.sum(betas[:,-1])
        betas[:,-1] = betas[:,-1] / c
        llBackward = torch.log(c)

        for t in range(T-2,-1,-1):
            start = max(0,L-2*(T-t))
            end = min(2*t+2,L)
            for s in range(end-1,-1,-1):
                l = int((s-1)/2)
                # blank
                if s%2 == 0:
                    if s == L-1:
                        betas[s,t] = betas[s,t+1] * params[blank,t]
                    else:
                        betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[blank,t]
                # same label twice
                elif s == L-2 or seq[l] == seq[l+1]:
                    # betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t] * (1 + cosdist_for_ctc[l])
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t]
                else:
                    # betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) * params[seq[l],t] * (1 + cosdist_for_ctc[l])
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) * params[seq[l],t]

            # normalize at current time
            c = torch.sum(betas[start:end,t])
            betas[start:end,t] = betas[start:end,t] / c
            llBackward = llBackward + torch.log(c)

        ctx.save_for_backward(params, seq, alphas, betas, llForward, llBackward)

        # ctc_loss_mean = -llForward / seqLen

        # return ctc_loss_mean

        return -llForward

    @staticmethod
    def backward(ctx, grad_output):
        params, seq, alphas, betas, llForward, llBackward = ctx.saved_tensors
        blank = 0
        seqLen = seq.shape[0]  # length of label sequence
        # seqLen = len(seq)
        L = 2*seqLen + 1  # length of the label sequence with blanks
        numphones = params.shape[0]  # number of labels
        T = params.shape[1]  # length of utterance (time)

        # =============================
        # STANFORD-CTC GRAD CALCULATION
        # Compute gradient with respect to unnormalized input parameters
        # grad = torch.zeros(params.shape, device=device).double()
        # ab = alphas*betas
        # for s in range(L):
        #     l = int((s-1)/2)
        #     # blank
        #     if s%2 == 0:
        #         grad[blank,:] = grad[blank,:] + ab[s,:]
        #         ab[s,:] = ab[s,:]/params[blank,:]
        #     else:
        #         grad[seq[l],:] = grad[seq[l],:] + ab[s,:]
        #         ab[s,:] = ab[s,:]/(params[seq[l],:])
        # absum = torch.sum(ab,axis=0)

        # # Check for underflow or zeros in denominator of gradient
        # llDiff = torch.abs(llForward-llBackward)
        # if llDiff > 1e-5 or torch.sum(absum==0) > 0:
        #     transposed_grad = grad.transpose(1,0)
        #     return (transposed_grad, None, None)
        # else:
        #     grad = params - grad / (params * absum)
        #     transposed_grad = grad.transpose(1,0)
        #     return (transposed_grad, None, None)

        # ============= ctc_fast.pyx grad implementation =============
        grad = torch.zeros(params.shape, device=device)
        ab = alphas*betas

        for s in range(L):
            # blank
            if s%2 == 0:
                for t in range(T):
                    grad[blank,t] += ab[s,t]
                    if ab[s,t] != 0:
                        ab[s,t] = ab[s,t]/params[blank,t]
            else:
                for t in range(T):
                    k = int((s-1)/2)
                    grad[seq[k],t] += ab[s,t]
                    if ab[s,t] != 0:
                        ab[s,t] = ab[s,t]/(params[seq[k],t])

        absum = torch.sum(ab, axis=0)

        grad = params - grad / (params * absum)
        # for t in range(T):
        #     for s in range(numphones):
        #         tmp = (params[s,t]*absum[t])
        #         if tmp > 0:
        #             grad[s,t] = params[s,t] - grad[s,t] / tmp
        #         else:
        #             grad[s,t] = params[s,t]

        return (grad.transpose(1,0), None)

        # =============================
        # NUMPY-CTC GRAD CALCULATION
        # padded_labels = torch.zeros((L))
        # j = 0
        # for i in range(L):
        #     if i%2 == 0:
        #         padded_labels[i] = blank
        #     else:
        #         padded_labels[i] = seq[j]
        #         j += 1

        # grad = torch.zeros(params.shape, device=device).double()

        # score_last = alphas[L-1, T-1]
        # score_before_last = betas[L-2, T-1]
        # p_l_given_ctc = score_last + score_before_last

        # for t in range(T):
        #     for k in range(numphones):
        #         d_p_d_ytk = 0
        #         lab_lk = np.nonzero(list(map(lambda x: 1 if k in x else 0, padded_labels)))[0]
        #         for s in lab_lk:
        #             d_p_d_ytk += alphas[s, t] * betas[s, t]

        #         d_p_d_ytk /= (params[k, t] ** 2)
        #         # d_lnp_d_ytk = (1. / p_l_given_ctc) * d_p_d_ytk
        #         # grad[k, t] = d_lnp_d_ytk
        #         grad[k, t] = d_p_d_ytk

        # return (grad.transpose(1,0), None, None)



# USING STANF0RD-CTC CODE:

def compute_CTCloss_withASD(reference_text, predicted_text, ref_label_ids, output_logits):  # originally includes: asd_model, asd_tokenizer
    # loss = torch.zeros((1), requires_grad=True, device=device).double()
    loss = torch.zeros((len(reference_text)), requires_grad=True, device=device).double()
    for i in range(len(reference_text)):
        ref_text = reference_text[i].replace("[UNK]", "")
        pred_text = predicted_text[i].replace("[UNK]", "")
        label_ids = ref_label_ids[i]
        labels_mask = label_ids >= 0
        flattened_labels = label_ids.masked_select(labels_mask)
        logits = output_logits[i]
        # ref_alignments = get_asd_align(ref_text, pred_text, asd_model, asd_tokenizer)
        # tokens_compressed = get_per_token_cosdist(ref_alignments)
        # cosdist_for_ctc = get_cosdist_for_ctc(tokens_compressed, flattened_labels)
        myctcloss = MyCTC.apply
        # custom_loss = myctcloss(logits, flattened_labels, cosdist_for_ctc)
        # loss = loss + custom_loss
        # loss[i] = myctcloss(logits, flattened_labels, cosdist_for_ctc)
        loss[i] = myctcloss(logits, flattened_labels)
        # print("custom loss:", custom_loss, "accumulated loss:", loss)
    # loss = loss / len(reference_text)
    # print("BATCH LOSS:", loss.sum())
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