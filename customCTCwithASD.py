import numpy as np
import torch
import re
from typing import Any, Dict, List, Optional, Union, Literal, Tuple
from dtw import *
import torch
import torch.nn.functional as F
from scipy.spatial import distance
import ctc_optimized  # cython built ctc loss & grad calc
import ctc  # python-implemented ctc loss & grad calc
import asd_for_ctc  # to extract ASD metric aligned to label seq for CTC loss calc
from jiwer import wer

# from torchaudio.models.decoder import ctc_decoder

# import k2
# import k2.ragged as k2r

from customloss_utils import (
    IGNORE_ID,
    MIN_LOG_VAL,
    make_pad_mask,
    mask_finished_preds,
    mask_finished_scores
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def compute_asd_score_single_utt(model, tokenizer, reference, hypothesis, normalized=True):
    ref_text = re.sub(r"\s+", " ", reference.replace("[UNK]", ""))
    hyp_text = re.sub(r"\s+", " ", hypothesis.replace("[UNK]", "").replace("</s>", ""))
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
    if normalized == True:
        asd_score = alignment.distance / num_tokens
    else:
        asd_score = alignment.distance
    # print("ref_text:", ref_text)
    # print("hyp_text:", hyp_text)
    # print("asd_score:", asd_score)
    return asd_score


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
    return torch.sum(log_indexes_probs, dim=0) / (model_pred_length.unsqueeze(-1))  # torch.sum dim=0 (original)


def get_paths(sampled_logits):
    paths_list = []
    index_list = torch.argmax(sampled_logits, dim=1)
    for idx in index_list:
        paths_list.append([idx])
    return torch.tensor(paths_list).type(torch.LongTensor).to(device)




# INCORPORATING ASD COSDIST VALUES TO THE CTC CALCULATION
# and defining it as a custom autograd function
# class MyCTC(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, logits, seq, input_length, cosdist_for_ctc):

#         # ============= CYTHON CTC loss implementation =============
#         params = logits.transpose(1,0)
#         # convert logits to log probs
#         params = params - (torch.max(params, dim=0)[0])
#         params = torch.exp(params)
#         params = params / torch.sum(params, dim=0)

#         params_arr = params.double().detach().cpu().numpy()
#         seq_arr = seq.int().detach().cpu().numpy()
#         cosdist_arr = np.array(cosdist_for_ctc, dtype=np.float64)
#         # llForward, llBackward, alphas, betas = ctc_optimized.forward_pass(params_arr, seq_arr, blank=31)
#         llForward, llBackward, alphas, betas = ctc_optimized.forward_pass_with_ASD(params_arr, seq_arr, cosdist_arr, blank=31)

#         alphas_tensor = torch.from_numpy(alphas).to(device)
#         betas_tensor = torch.from_numpy(betas).to(device)
#         llForward_tensor = torch.tensor(llForward).to(device)
#         llBackward_tensor = torch.tensor(llBackward).to(device)

#         ctx.save_for_backward(params, seq, input_length, alphas_tensor, betas_tensor, llForward_tensor, llBackward_tensor)

#         return llForward_tensor

#         # # ============= ctc loss implementation =============
#         # llForward, llBackward, alphas, betas, params = ctc.forward_pass(logits, seq, device, blank=31)
#         # ctx.save_for_backward(params, seq, input_length, alphas, betas, llForward, llBackward)

#         # return llForward

#     @staticmethod
#     def backward(ctx, grad_output):

#         # ============= CYTHON grad implementation =============
#         params, seq, input_length, alphas, betas, llForward, llBackward = ctx.saved_tensors
#         params_arr = params.double().detach().cpu().numpy()
#         seq_arr = seq.int().detach().cpu().numpy()
#         alphas_arr = alphas.double().detach().cpu().numpy()
#         betas_arr = betas.double().detach().cpu().numpy()
#         input_len_int = input_length.int().detach().cpu()

#         grad = ctc_optimized.backward_pass(params_arr, seq_arr, alphas_arr, betas_arr, input_len_int, blank=31)

#         grad_tensor = torch.tensor(grad).to(device)

#         return (grad_tensor.transpose(1,0), None, None, None, None)

#         # # ============= ctc grad implementation =============
#         # params, seq, input_length, alphas, betas, llForward, llBackward = ctx.saved_tensors
#         # grad = ctc.backward_pass(params, seq, alphas, betas, device, blank=31)

#         # return (grad.transpose(1,0), None, None)


# # USING STANF0RD-CTC CODE:
# def compute_CTCloss_withASD(reference_text, predicted_text, ref_label_ids, output_logits, input_lengths, asd_model, asd_tokenizer):  # originally includes: asd_model, asd_tokenizer
#     loss = torch.zeros((len(reference_text)), requires_grad=True, device=device).double()
#     for i in range(len(reference_text)):
#         ref_text = reference_text[i].replace("[UNK]", "")
#         pred_text = predicted_text[i].replace("[UNK]", "")
#         label_ids = ref_label_ids[i]
#         labels_mask = label_ids >= 0
#         flattened_labels = label_ids.masked_select(labels_mask)
#         logits = output_logits[i]
#         ref_alignments = asd_for_ctc.get_asd_align(ref_text, pred_text, asd_model, asd_tokenizer)
#         tokens_compressed = asd_for_ctc.get_per_token_cosdist(ref_alignments)
#         cosdist_for_ctc = asd_for_ctc.get_cosdist_for_ctc(tokens_compressed, flattened_labels)
#         myctcloss = MyCTC.apply
#         if len(flattened_labels) != len(cosdist_for_ctc):
#             raise Exception("cosdist for ctc length not equal to flattened labels length")
#         loss[i] = myctcloss(logits, flattened_labels, input_lengths[i], cosdist_for_ctc)
#         # loss[i] = myctcloss(logits, flattened_labels, input_lengths[i])
#         # print("loss:", loss[i], "lambda:", lambda_asd)
#         # print("ref:", ref_text)
#         # print("hyp:", pred_text)
#         # print("loss:", loss[i])
#     return loss.sum()


# def compute_CTCloss_nbest(reference_text, output_logits, input_lengths, asd_model, asd_tokenizer):
#     decoder = ctc_decoder(lexicon=None, tokens="tokens.txt", nbest=10, beam_size=100, blank_token="[PAD]",
#                           sil_token="|", unk_word="[UNK]")
#     targets = []
#     target_lengths = []
#     log_probs = F.log_softmax(output_logits, dim=-1, dtype=torch.float32).transpose(0, 1)

#     for i in range(len(reference_text)):
#         ref_text = reference_text[i].replace("[UNK]", "")
#         logits = output_logits[i]
#         # get nbest hypotheses and rank them
#         nbest_list = decoder(logits.type(torch.float32).detach().cpu()[None, :, :])
#         nbest_token_list = []
#         asd_score_list = [0] * len(nbest_list[0])
#         hyp_list = []
#         for j, item in enumerate(nbest_list[0]):
#             tokens = item.tokens
#             for k in range(len(tokens)):
#                 if tokens[k] == 0:
#                     tokens_mod = tokens[k+1:]
#                 else:
#                     break
#             chars = decoder.idxs_to_tokens(tokens_mod)
#             nbest_token_list.append(tokens_mod)
#             hyp_text = re.sub(" +", " ", "".join(chars).replace("|", " "))
#             hyp_list.append(hyp_text)
#             asd_score_list[j] = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text, hyp_text)
#         targets.append(torch.tensor(nbest_token_list[np.argmin(asd_score_list)]))
#         target_lengths.append(torch.tensor(len(nbest_token_list[np.argmin(asd_score_list)])))

#     targets_tensor = torch.cat(targets, dim=0)
#     targets_len_tensor = torch.tensor(target_lengths)

#     with torch.backends.cudnn.flags(enabled=False):
#         loss = F.ctc_loss(
#                 log_probs,
#                 targets_tensor,
#                 input_lengths,
#                 targets_len_tensor,
#                 blank=31,
#                 reduction="mean",
#                 zero_infinity=True,
#                 )

#     return loss


# # this one did not affect the model outputs i think
# def compute_nbest_asd(reference_text, output_logits, input_lengths, asd_model, asd_tokenizer):
#     decoder = ctc_decoder(lexicon=None, tokens="tokens.txt", nbest=10, beam_size=100, blank_token="[PAD]",
#                           sil_token="|", unk_word="[UNK]")
#     loss = torch.zeros((len(reference_text)), requires_grad=True, device=device).double()
#     # log_probs = F.log_softmax(output_logits, dim=-1, dtype=torch.float32)
#     for i in range(len(reference_text)):
#         ref_text = reference_text[i].replace("[UNK]", "")
#         logits = output_logits[i]
#         # get nbest hypotheses and get path log probs * asd score
#         nbest_list = decoder(logits.type(torch.float32).detach().cpu()[None, :, :])
#         nbest_asd_loss = torch.zeros((len(nbest_list[0])), requires_grad=True, device=device).double()
#         for j, item in enumerate(nbest_list[0]):
#             path_probs = torch.zeros((len(item.tokens)), requires_grad=True, device=device).double()
#             chars = decoder.idxs_to_tokens(item.tokens)
#             hyp_text = re.sub(" +", " ", "".join(chars).replace("|", " "))
#             for k, token in enumerate(item.tokens):
#                 print(item.score)
#                 if item.timesteps[k] < input_lengths[i]:
#                     start = item.timesteps[k]
#                     if k < (len(item.timesteps) - 1):
#                         end = item.timesteps[k+1]
#                     else:
#                         end = item.timesteps[-1]
#                     path_probs[k] = torch.clamp(logits[start:end, token], min=0).sum()
#                     # path_probs[k] = logits[start:end, token].sum()
#             nbest_asd_loss[j] = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text, hyp_text) * (path_probs.sum() / input_lengths[i])
#         loss[i] = nbest_asd_loss.mean()
#     return loss.mean()




# # ===================================================
# # CTC with Gumbel-Softmax sampled ASD scoring

# def sampled_logits_asd_loss(reference_text, predicted_text, output_logits, metric_model, metric_tokenizer):
#     # calculate asd scores for batch:
#     asd_scores = compute_asd_score_batch(metric_model, metric_tokenizer, reference_text, predicted_text)
#     # sample from output logits:
#     sampled_logits = F.gumbel_softmax(output_logits, tau=1, hard=True, dim=-1)
#     # sampled logits x ASD score:
#     temp_list = []
#     for i, logits in enumerate(sampled_logits):
#         asd_matrix = torch.full_like(logits, (1 + (asd_scores[i]*10)), requires_grad=False)
#         temp_list.append(logits.detach() * asd_matrix)
#     sampled_logits_asd = torch.stack(temp_list, dim=0)
#     sampled_logits_asd.unsqueeze(0)
#     # calculate loss:
#     # L1loss = torch.nn.SmoothL1Loss(reduction="mean", beta=1)
#     L1loss = torch.nn.L1Loss(reduction="mean")
#     loss = L1loss(input=sampled_logits, target=sampled_logits_asd) * 10
#     # MSEloss = torch.nn.MSELoss(reduction="mean")
#     # loss = MSEloss(input=sampled_logits, target=sampled_logits_asd) * 100
#     return loss


# def compute_sampled_meanASD(reference_text, output_logits, asd_model, asd_tokenizer, processor):
#     num_samples = 5
#     loss = torch.zeros((len(reference_text)), requires_grad=True, device=device).double()
#     for i in range(len(reference_text)):
#         ref_text = reference_text[i].replace("[UNK]", "")
#         logits = output_logits[i]
#         asd_scores = [0] * num_samples
#         # get samples using gumble softmax sampling
#         for j in range(num_samples):
#             sampled_logits = F.gumbel_softmax(logits, tau=10, hard=True, dim=-1)
#             hyp_text = processor.decode(sampled_logits.detach().cpu().numpy()).text
#             asd_scores[j] = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text, hyp_text)
#         loss[i] = np.mean(asd_scores)
#     return torch.mean(loss)


# def sampled_pair_hinge_loss(ref_text, output_logits, input_lengths, asd_model, asd_tokenizer, processor):
#     loss = torch.zeros((len(ref_text)), requires_grad=True, device=device).double()
#     for i in range(len(ref_text)):
#         log_probs = F.log_softmax(output_logits[i], dim=-1, dtype=torch.float32)
#         # non_zero_logits = []
#         paths_list = []
#         non_zero_asd = []
#         while len(non_zero_asd) < 2:
#             sampled_logits = F.gumbel_softmax(output_logits[i], tau=100, hard=True, dim=-1)
#             hyp_text = processor.decode(sampled_logits.detach().cpu().numpy()).text
#             asd_score = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text[i], hyp_text)
#             if asd_score != 0:
#                 # non_zero_logits.append(sampled_logits)
#                 paths_list.append(get_paths(sampled_logits))
#                 non_zero_asd.append(asd_score)
#         # mass_prob_0 = get_mass_prob(non_zero_logits[0].type(torch.LongTensor).to(device), log_probs, input_lengths[i])
#         # mass_prob_1 = get_mass_prob(non_zero_logits[1].type(torch.LongTensor).to(device), log_probs, input_lengths[i])
#         mass_prob_0 = get_mass_prob(paths_list[0], log_probs, input_lengths[i])
#         mass_prob_1 = get_mass_prob(paths_list[1], log_probs, input_lengths[i])
#         if non_zero_asd[0] < non_zero_asd[1]:
#             subtract_path_probs = mass_prob_1 - mass_prob_0
#         else:
#             subtract_path_probs = mass_prob_0 - mass_prob_1
#         # loss[i] = torch.sum(torch.clamp(subtract_path_probs, min=0))
#         loss[i] = torch.clamp(subtract_path_probs, min=0)
#     return torch.mean(loss)


# def sampled_multi_hinge_loss(ref_text, output_logits, input_lengths, asd_model, asd_tokenizer, processor):
#     num_samples = 10
#     loss = torch.zeros((len(ref_text)), requires_grad=True, device=device).double()
#     for i in range(len(ref_text)):
#         log_probs = F.log_softmax(output_logits[i], dim=-1, dtype=torch.float32)
#         non_zero_logits = []
#         non_zero_asd = []
#         while len(non_zero_asd) < num_samples:
#             sampled_logits = F.gumbel_softmax(output_logits[i], tau=100, hard=True, dim=-1)
#             hyp_text = processor.decode(sampled_logits.detach().cpu().numpy()).text
#             asd_score = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text[i], hyp_text)
#             if asd_score != 0:
#                 non_zero_logits.append(sampled_logits)
#                 non_zero_asd.append(asd_score)
#         lowest_asd_idx = np.argmin(np.array(non_zero_asd))
#         mass_prob_list = []
#         for j in range(len(non_zero_asd)):
#             mass_prob_list.append(get_mass_prob(non_zero_logits[j].type(torch.LongTensor).to(device), log_probs, input_lengths[i]))
#         local_loss = torch.zeros((len(mass_prob_list)-1), requires_grad=True, device=device).double()
#         l = 0
#         for k in range(len(mass_prob_list)):
#             if k != lowest_asd_idx:
#                 subtract_path_probs = mass_prob_list[k] - mass_prob_list[lowest_asd_idx]
#                 local_loss[l] = torch.sum(torch.clamp((subtract_path_probs), min=0))
#                 l += 1
#         loss[i] = torch.mean(local_loss)
#     return torch.mean(loss)


# def sampled_multi_hinge_loss_ver2(ref_text, output_logits, input_lengths, asd_model, asd_tokenizer, processor):
#     num_samples = 10
#     loss = torch.zeros((len(ref_text)), requires_grad=True, device=device).double()
#     for i in range(len(ref_text)):
#         log_probs = F.log_softmax(output_logits[i], dim=-1, dtype=torch.float32)
#         non_zero_logits = []
#         non_zero_asd = []
#         while len(non_zero_asd) < num_samples:
#             sampled_logits = F.gumbel_softmax(output_logits[i], tau=100, hard=True, dim=-1)
#             hyp_text = processor.decode(sampled_logits.detach().cpu().numpy()).text
#             asd_score = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text[i], hyp_text)
#             non_zero_logits.append(sampled_logits)
#             non_zero_asd.append(asd_score)
#         lowest_asd_idx = np.argmin(np.array(non_zero_asd))
#         mass_prob_list = []
#         for j in range(len(non_zero_asd)):
#             mass_prob_list.append(get_mass_prob(non_zero_logits[j].type(torch.LongTensor).to(device), log_probs, input_lengths[i]))
#         local_loss = torch.zeros((len(mass_prob_list)-1), requires_grad=True, device=device).double()
#         l = 0
#         for k in range(len(mass_prob_list)):
#             if k != lowest_asd_idx:
#                 subtract_path_probs = mass_prob_list[k] - mass_prob_list[lowest_asd_idx]
#                 local_loss[l] = torch.sum(torch.clamp((subtract_path_probs), min=0))
#                 l += 1
#         loss[i] = torch.mean(local_loss)
#     return torch.mean(loss)


# # get mass probability of sampled path & calculate the expected ASD score

# # this function doesn't seem to change the eval wer/asd with each iteration, no effect to the model weights?
# def sampled_multi_expected_ASD(ref_text, output_logits, input_lengths, asd_model, asd_tokenizer, processor):
#     num_samples = 5
#     loss = torch.zeros((len(ref_text)), requires_grad=True, device=device).double()
#     for i in range(len(ref_text)):
#         log_probs = F.log_softmax(output_logits[i], dim=-1, dtype=torch.float32)
#         samples_loss = torch.zeros(num_samples, requires_grad=True, device=device).double()
#         for j in range(num_samples):
#             sampled_logits = F.gumbel_softmax(output_logits[i], tau=10, hard=True, dim=-1)
#             hyp_text = processor.decode(sampled_logits.detach().cpu().numpy()).text
#             asd_score = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text[i], hyp_text)
#             mass_prob = get_mass_prob(sampled_logits.type(torch.LongTensor).to(device), log_probs, input_lengths[i])
#             mass_prob_sum = torch.sum(mass_prob)
#             path_prob = mass_prob_sum * sampled_logits
#             path_prob_norm = torch.exp(path_prob / torch.sum(path_prob)) * asd_score
#             samples_loss[j] = torch.mean(path_prob_norm)
#             # samples_loss[j] = torch.sum(torch.clamp(mass_prob, min=0)) * asd_score
#         loss[i] = torch.mean(samples_loss)
#     return torch.mean(loss)


# def sampled_multi_expected_ASD_ver2(ref_text, output_logits, input_lengths, asd_model, asd_tokenizer, processor):
#     num_samples = 5
#     loss = torch.zeros((len(ref_text)), requires_grad=True, device=device).double()
#     for i in range(len(ref_text)):
#         log_probs = F.log_softmax(output_logits[i], dim=-1, dtype=torch.float32)
#         samples_loss = torch.zeros(num_samples, requires_grad=True, device=device).double()
#         sample_probs = torch.zeros(num_samples, requires_grad=True, device=device).double()
#         asd_scores = []
#         for j in range(num_samples):
#             sampled_logits = F.gumbel_softmax(output_logits[i], tau=10, hard=True, dim=-1)
#             hyp_text = processor.decode(sampled_logits.detach().cpu().numpy()).text
#             asd_scores.append(compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text[i], hyp_text))
#             mass_prob = get_mass_prob(sampled_logits.type(torch.LongTensor).to(device), log_probs, input_lengths[i])
#             sample_probs[j] = torch.sum(mass_prob)
#         for k in range(num_samples):
#             samples_loss[k] = torch.exp((sample_probs[k] / torch.sum(sample_probs)) * asd_scores[k])
#         loss[i] = torch.mean(samples_loss)
#     return torch.mean(loss)


# def sampled_multi_expected_ASD_ver3(ref_text, output_logits, input_lengths, asd_model, asd_tokenizer, processor):
#     num_samples = 5
#     loss = torch.zeros((len(ref_text)), requires_grad=True, device=device).double()
#     for i in range(len(ref_text)):
#         log_probs = F.log_softmax(output_logits[i], dim=-1, dtype=torch.float32)
#         samples_loss = torch.zeros(num_samples, requires_grad=True, device=device).double()
#         for j in range(num_samples):
#             sampled_logits = F.gumbel_softmax(output_logits[i], tau=10, hard=True, dim=-1)
#             # new definition of the path:
#             paths_tensor = get_paths(sampled_logits)
#             hyp_text = processor.decode(sampled_logits.detach().clone().cpu().numpy()).text
#             asd_score = torch.tensor(compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text[i], hyp_text),
#                                      device=device, requires_grad=True)
#             # mass_prob = get_mass_prob(sampled_logits.type(torch.LongTensor).to(device), log_probs, input_lengths[i])
#             mass_prob = get_mass_prob(paths_tensor, log_probs, input_lengths[i])
#             samples_loss[j] = torch.exp(mass_prob) * (asd_score)
#         loss[i] = torch.mean(samples_loss)
#     return torch.mean(loss)




# ===================================================
# CE-OptimizedLoss MWER
# source: https://github.com/TeaPoly/CE-OptimizedLoss/blob/master/mwer.py

def create_sampling_mask(log_softmax, n):
    """
    Generate sampling mask

    # Ref: <Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition>
    #       https://arxiv.org/abs/2206.08317

    Args:
        log_softmax: log softmax inputs, float32 (batch, maxlen_out, vocab_size)
        n: candidate paths num, int32
    Return:
        sampling_mask: the sampling mask (nbest, batch, maxlen_out, vocab_size)
    """
    b, s, v = log_softmax.size()

    # Generate random mask
    nbest_random_mask = torch.randint(
        0, 2, (n, b, s, v), device=log_softmax.device)

    # Greedy search decoding for best path
    top1_score_indices = log_softmax.argmax(dim=-1).squeeze(-1)

    # Genrate top 1 score token mask
    top1_score_indices_mask = torch.zeros((b, s, v), dtype=torch.int).to(
        log_softmax.device
    )
    top1_score_indices_mask.scatter_(-1, top1_score_indices.unsqueeze(-1), 1)

    # Genrate sampling mask by applying random mask to top 1 score token
    sampling_mask = nbest_random_mask * top1_score_indices_mask.unsqueeze(0)

    return sampling_mask


def negative_sampling_decoder(
    logit: torch.Tensor,
    nbest: int = 4,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate multiple candidate paths by negative sampling strategy

    # Ref: <Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition>
    #       https://arxiv.org/abs/2206.08317

    Args:
        logit: logit inputs, float32 (batch, maxlen_out, vocab_size)
        nbest: candidate paths num, int32
        masks: logit lengths, (batch, maxlen_out)
    Return:
        nbest_log_distribution: the N-BEST distribution of candidate path (nbest, batch)
        nbest_pred: the NBEST candidate path (nbest, batch, maxlen_out)
    """

    # Using log-softmax for probability distribution
    log_softmax = torch.nn.functional.log_softmax(logit, dim=-1)

    # Generate sampling mask
    with torch.no_grad():
        sampling_mask = create_sampling_mask(log_softmax, nbest)

    # Randomly masking top1 score with -float('inf')
    # (nbest, batch, maxlen_out, vocab_size)
    nbest_log_softmax = torch.where(
        sampling_mask != 0, MIN_LOG_VAL.type_as(log_softmax), log_softmax
    )

    # Greedy search decoding for sampling log softmax
    nbest_logsoftmax, nbest_pred = nbest_log_softmax.topk(1)
    nbest_pred = nbest_pred.squeeze(-1)
    nbest_logsoftmax = nbest_logsoftmax.squeeze(-1)

    # Construct N-BEST log PDF
    # FIXME (huanglk): Ignore irrelevant probabilities
    # (n, b, s) -> (n, b): log(p1*p2*...pn) = log(p1)+log(p2)+...log(pn)
    nbest_log_distribution = torch.sum(
        nbest_logsoftmax.masked_fill(masks, 0), -1)

    return nbest_log_distribution, nbest_pred


def batch_beam_search(
    logit: torch.Tensor, beam_size: int, masks: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Beam Search Decoder

    Parameters:

        logit(Tensor): the logit of network.
        beam_size(int): beam size of decoder.

    Outputs:

        indices(Tensor): a beam of index sequence.
        log_prob(Tensor): a beam of log likelihood of sequence.

    Shape:

        post: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).

    Examples:

        >>> post = torch.softmax(torch.randn([32, 20, 1000]), -1)
        >>> indices, log_prob = beam_search_decoder(post, 3)

    """
    batch_size, seq_length, vocab_size = logit.shape
    eos = vocab_size - 1
    # beam search
    with torch.no_grad():
        # b,t,v
        log_post = torch.nn.functional.log_softmax(logit, dim=-1)
        # b,k
        log_prob, indices = log_post[:, 0, :].topk(beam_size, sorted=True)
        end_flag = torch.eq(masks[:, 0], 1).view(-1, 1)
        # mask predictor and scores if end
        log_prob = mask_finished_scores(log_prob, end_flag)
        indices = mask_finished_preds(indices, end_flag, eos)
        # b,k,1
        indices = indices.unsqueeze(-1)

        for i in range(1, seq_length):
            # b,v
            scores = mask_finished_scores(log_post[:, i, :], end_flag)
            # b,v -> b,k,v
            topk_scores = scores.unsqueeze(1).repeat(1, beam_size, 1)
            # b,k,1 + b,k,v -> b,k,v
            top_k_logp = log_prob.unsqueeze(-1) + topk_scores

            # b,k,v -> b,k*v -> b,k
            log_prob, top_k_index = top_k_logp.view(batch_size, -1).topk(
                beam_size, sorted=True
            )

            index = mask_finished_preds(top_k_index, end_flag, eos)

            indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)

            end_flag = torch.eq(masks[:, i], 1).view(-1, 1)

        indices = torch.fmod(indices, vocab_size)
    return indices, log_prob


def beam_search_decoder(
    logit: torch.Tensor, beam_size: int, masks: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Beam Search Decoder

    Parameters:

        logit(Tensor): the logit of network.
        beam_size(int): beam size of decoder.

    Outputs:

        indices(Tensor): a beam of index sequence.
        log_prob(Tensor): a beam of log likelihood of sequence.

    Shape:

        post: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).

    Examples:

        >>> post = torch.softmax(torch.randn([32, 20, 1000]), -1)
        >>> indices, log_prob = beam_search_decoder(post, 3)

    """
    # beam search decoder
    indices, _ = batch_beam_search(logit, beam_size, masks)
    # recompute PDF for gradient
    log_post = torch.nn.functional.log_softmax(logit, dim=-1)
    # b,t,v -> b,n,t,v
    nlog_post = log_post.unsqueeze(1).repeat(1, beam_size, 1, 1)
    # indices: b, n, t -> b, n, t
    top_k_log_post = torch.gather(
        nlog_post, -1, indices.unsqueeze(-1)).squeeze(-1)
    # b, n, t -> b, n
    topk_log_prob = torch.sum(
        top_k_log_post.masked_fill(masks.unsqueeze(1), 0), -1)
    return topk_log_prob.transpose(0, 1), indices.transpose(0, 1)


def beam_search_decoder_mod(logit: torch.Tensor, beam_size: int, masks: torch.Tensor):
    # beam search decoder
    indices, _ = batch_beam_search(logit, beam_size, masks)
    # print("indices:", indices.shape, indices.requires_grad)

    # recompute PDF for gradient
    log_post = torch.nn.functional.log_softmax(logit, dim=-1)
    # print("log_post:", log_post.shape, log_post.requires_grad)

    # b,t,v -> b,n,t,v
    nlog_post = log_post.unsqueeze(1).repeat(1, beam_size, 1, 1)
    # print("nlog_post:", nlog_post.shape, nlog_post.requires_grad)

    # indices: b, n, t -> b, n, t
    top_k_log_post = torch.gather(nlog_post, -1, indices.unsqueeze(-1)).squeeze(-1)
    # print("top_k_log_post:", top_k_log_post.shape, top_k_log_post.requires_grad)

    # b, n, t -> b, n
    topk_log_prob = torch.sum(top_k_log_post.masked_fill(masks.unsqueeze(1), 0), -1)
    # print("topk_log_prob:", topk_log_prob.shape, topk_log_prob.requires_grad)

    # MOD: extracting log probs for hyp decoding
    # nlog_probs = torch.zeros_like(nlog_post)
    # batch_size = nlog_probs.size()[0]
    # num_frames = nlog_probs.size()[2]
    # num_samples = beam_size
    # for i in range(batch_size):
    #     for j in range(num_samples):
    #         for k in range(num_frames):
    #             nlog_probs[i,j,k,indices[i,j,k]] = 1.0

    # return topk_log_prob.transpose(0, 1), indices.transpose(0, 1), nlog_probs.transpose(0, 1)
    return topk_log_prob.transpose(0, 1), indices.transpose(0, 1)


def compute_mwer_loss(
    nbest_log_distribution=torch.Tensor,
    nbest_pred=torch.Tensor,
    tgt=torch.Tensor,
    masks=torch.Tensor,
):
    """
    Compute Minimum Word Error Rate Training loss.

    # Ref: <Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models>
    #       https://arxiv.org/abs/1712.01818

    Args:
        nbest_log_distribution: the N-BEST distribution of candidate path (nbest, batch)
        nbest_pred: the NBEST candidate path (nbest, batch, maxlen_out)
        tgt: padded target token ids, int32 (batch, maxlen_out)
        masks: target token lengths of this batch (batch,)
    Return:
        loss: normalized MWER loss (batch,)
    """
    n, b, s = nbest_pred.size()

    # necessary to filter irrelevant length
    # (b,) -> (b, s)
    # not include <eos/sos>
    tgt = tgt.masked_fill(masks, IGNORE_ID)
    # (n, b, s)
    nbest_pred = nbest_pred.masked_fill(masks, IGNORE_ID)

    # Construct number of word errors
    # (b, s) -> (n, b, s)
    tgt = tgt.unsqueeze(0).repeat(n, 1, 1)

    # convert to float for normalize
    # (n, b, s) -> (n, b)
    nbest_word_err_num = torch.sum((tgt != nbest_pred), -1).float()

    # Computes log distribution
    # (n, b) -> (b,): log( p1+p2+...+pn ) = log( exp(log_p1)+exp(log_p2)+...+exp(log_pn) )
    sum_nbest_log_distribution = torch.logsumexp(nbest_log_distribution, 0)

    # Re-normalized over just the N-best hypotheses.
    # (n, b) - (b,) -> (n, b): exp(log_p)/exp(log_p_sum) = exp(log_p-log_p_sum)
    normal_nbest_distribution = torch.exp(
        nbest_log_distribution - sum_nbest_log_distribution
    )

    # Average number of word errors over the N-best hypohtheses
    # (n, b) -> (b)
    mean_word_err_num = torch.mean(nbest_word_err_num, 0)
    # print("mean_word_err_num:", mean_word_err_num)

    # Re-normalized error word number over just the N-best hypotheses
    # (n, b) - (b,) -> (n, b)
    normal_nbest_word_err_num = nbest_word_err_num - mean_word_err_num

    # Expected number of word errors over the training set.
    # (n, b) -> (b,)
    mwer_loss = torch.sum(normal_nbest_distribution *
                          normal_nbest_word_err_num, 0)

    return mwer_loss


def compute_masd_loss(nbest_log_distribution, asd_scores):
    # Computes log distribution
    # (n, b) -> (b,): log( p1+p2+...+pn ) = log( exp(log_p1)+exp(log_p2)+...+exp(log_pn) )
    sum_nbest_log_distribution = torch.logsumexp(nbest_log_distribution, 0)

    # Re-normalized over just the N-best hypotheses.
    # (n, b) - (b,) -> (n, b): exp(log_p)/exp(log_p_sum) = exp(log_p-log_p_sum)
    normal_nbest_distribution = torch.exp(nbest_log_distribution - sum_nbest_log_distribution)

    # Average number of ASD score over the N-best hypohtheses
    # (n, b) -> (b)
    mean_asd = torch.mean(asd_scores, 0)

    # Re-normalized ASD scores over just the N-best hypotheses
    # (n, b) - (b,) -> (n, b)
    normalized_asd = asd_scores - mean_asd

    # Expected number of word errors over the training set.
    # (n, b) -> (b,)
    asd_loss = torch.sum(normal_nbest_distribution * normalized_asd, 0)

    return asd_loss


def compute_masd_loss_ver2(nbest_log_distribution, asd_scores, normalized_score=False):
    # Computes log distribution
    # (n, b) -> (b,): log( p1+p2+...+pn ) = log( exp(log_p1)+exp(log_p2)+...+exp(log_pn) )
    sum_nbest_log_distribution = torch.logsumexp(nbest_log_distribution, 0)
    # print("sum_nbest_log_distribution:", sum_nbest_log_distribution.shape, sum_nbest_log_distribution.requires_grad)

    # Re-normalized over just the N-best hypotheses.
    # (n, b) - (b,) -> (n, b): exp(log_p)/exp(log_p_sum) = exp(log_p-log_p_sum)
    normal_nbest_distribution = torch.exp(nbest_log_distribution - sum_nbest_log_distribution)
    # print("normal nbest dist:", normal_nbest_distribution.shape, normal_nbest_distribution.requires_grad)

    if normalized_score == True:
        mean_asd = torch.mean(asd_scores, 0)
        asd_norm = asd_scores - mean_asd
        asd_loss = torch.sum(normal_nbest_distribution * asd_norm, 0)
    else:
        asd_loss = torch.sum(normal_nbest_distribution * asd_scores, 0)
        # print("asd_loss:", asd_loss.shape, asd_loss.requires_grad)

    return asd_loss


# concept based on the align with purpose, but without shifting the path, just making the hinge loss smaller
# ranking of paths based on asd score
def compute_masd_loss_ver3(nbest_log_distribution, asd_scores, candidate_paths_num):
    mean_asd_scores = torch.mean(asd_scores, 1)
    min_idx = torch.argmin(mean_asd_scores)
    new_mean_asd_scores = torch.cat([mean_asd_scores[0:min_idx], mean_asd_scores[min_idx+1:]])
    new_nbest_log_distribution = torch.cat([nbest_log_distribution[0:min_idx,:],nbest_log_distribution[min_idx+1:,:]])
    new_min_idx = torch.argmin(new_mean_asd_scores)
    min_asd_log_dist = new_nbest_log_distribution[new_min_idx].repeat(candidate_paths_num - 1, 1)
    subtract_pairs = new_nbest_log_distribution - min_asd_log_dist
    loss = torch.clamp(subtract_pairs, min=0)
    print("loss:", loss)
    return loss


class Seq2seqMwerLoss(torch.nn.Module):
    """Minimum Word Error Rate Training loss based on the negative sampling strategy

    <Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition>
        https://arxiv.org/abs/2206.08317

    <Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models>
        https://arxiv.org/abs/1712.01818

    Args:
        candidate_paths_num (int): The number of candidate paths.
    """

    def __init__(
        self,
        sampling_method="beam_search",  # beam_search or negative_sampling
        candidate_paths_num: int = 4,
        reduction: str = "mean",
        eos_id: int = -1
    ):
        super().__init__()
        self.candidate_paths_num = candidate_paths_num
        self.sampling_method = sampling_method
        self.reduction = reduction
        self.eos_id = eos_id

    def forward(
        self, logit: torch.Tensor, tgt: torch.Tensor, tgt_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logit: logit (batch, maxlen_out, vocab_size)
            tgt: padded target token ids, int64 (batch, maxlen_out)
            tgt_lens: target lengths of this batch (batch)
        Return:
            loss: normalized MWER loss
        """
        assert tgt_lens.size()[0] == tgt.size()[0] == logit.size()[0]
        assert logit.size()[1] == tgt.size()[1]

        # not include <eos/sos>
        masks = make_pad_mask(
            tgt_lens if self.eos_id < 0 else tgt_lens - 1, max_len=logit.size()[1]
        )
        if self.sampling_method == "beam_search":
            # Beam search to generate multiple candidate paths
            nbest_log_distribution, nbest_pred = beam_search_decoder(
                logit, self.candidate_paths_num, masks
            )
        elif self.sampling_method == "negative_sampling":
            # Randomly mask the top1 score to generate multiple candidate paths
            nbest_log_distribution, nbest_pred = negative_sampling_decoder(
                logit, self.candidate_paths_num, masks
            )
        else:
            raise Exception(f"Not support sampling_method: {self.sampling_method} ")

        # Compute MWER loss
        mwer_loss = compute_mwer_loss(
            nbest_log_distribution, nbest_pred, tgt, masks)

        if self.reduction == "sum":
            return torch.sum(mwer_loss)
        elif self.reduction == "mean":
            return torch.mean(mwer_loss)
        else:
            return mwer_loss


class Seq2seqMASDLoss(torch.nn.Module):
    """Minimum Word Error Rate Training loss based on the negative sampling strategy

    <Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition>
        https://arxiv.org/abs/2206.08317

    <Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models>
        https://arxiv.org/abs/1712.01818

    Args:
        candidate_paths_num (int): The number of candidate paths.
    """

    def __init__(
        self,
        sampling_method="beam_search",  # beam_search or negative_sampling
        candidate_paths_num: int = 2,
        reduction: str = "mean",
        eos_id: int = 33
    ):
        super().__init__()
        self.candidate_paths_num = candidate_paths_num
        self.sampling_method = sampling_method
        self.reduction = reduction
        self.eos_id = eos_id


    def forward(
        self, nbest_log_distribution: torch.Tensor, ref_list, hyp_list, metric_model, metric_tokenizer, use_asd) -> torch.Tensor:
        """
        Args:
            logit: logit (batch, maxlen_out, vocab_size)
            tgt: padded target token ids, int64 (batch, maxlen_out)
            tgt_lens: target lengths of this batch (batch)
        Return:
            loss: normalized MWER loss
        """

        asd_scores = []
        for hyp_group in hyp_list:
            path_scores = []
            for ref, hyp in zip(ref_list, hyp_group):
                if use_asd == 1:
                    path_scores.append(compute_asd_score_single_utt(metric_model, metric_tokenizer, ref, hyp))
                elif use_asd == 0:
                    path_scores.append(wer(ref, hyp))
            asd_scores.append(path_scores)

        asd_scores_tensor = torch.tensor(asd_scores, device=device)

        # masd_loss = compute_masd_loss(nbest_log_distribution, asd_scores_tensor)
        masd_loss = compute_masd_loss_ver2(nbest_log_distribution, asd_scores_tensor, normalized_score=True)  # True for MWER trial
        # masd_loss = compute_masd_loss_ver3(nbest_log_distribution, asd_scores_tensor, self.candidate_paths_num)

        if self.reduction == "sum":
            return torch.sum(masd_loss)
        elif self.reduction == "mean":
            return torch.mean(masd_loss)
        else:
            return masd_loss


    def get_logits_for_decoding(self, logit: torch.Tensor, tgt_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logit: logit (batch, maxlen_out, vocab_size)
            tgt_lens: target lengths of this batch (batch)
        Return:
            nlog_probs: log probabilities for the paths (n_paths, batch, maxlen_out, vocab_size)
        """
        # not include <eos/sos>
        masks = make_pad_mask(
            tgt_lens if self.eos_id < 0 else tgt_lens - 1, max_len=logit.size()[1]
        )

        if self.sampling_method == "beam_search":
            # Beam search to generate multiple candidate paths
            # nbest_log_distribution, nbest_pred, nlog_probs = beam_search_decoder_mod(
            #     logit, self.candidate_paths_num, masks
            # )
            nbest_log_distribution, nbest_pred = beam_search_decoder_mod(
                logit, self.candidate_paths_num, masks
            )

        elif self.sampling_method == "negative_sampling":
            # Randomly mask the top1 score to generate multiple candidate paths
            nbest_log_distribution, nbest_pred = negative_sampling_decoder(
                logit, self.candidate_paths_num, masks
            )
        else:
            raise Exception(f"Not support sampling_method: {self.sampling_method} ")

        # return nbest_log_distribution, nlog_probs, nbest_pred
        return nbest_log_distribution, nbest_pred




# ===================================================
# k2 MWER Loss
# source: https://github.com/TeaPoly/CTC-OptimizedLoss/tree/main

# def get_lattice(
#     nnet_output: torch.Tensor,
#     decoding_graph: k2.Fsa,
#     supervision_segments: torch.Tensor,
#     search_beam: float,
#     output_beam: float,
#     min_active_states: int,
#     max_active_states: int,
#     subsampling_factor: int = 1,
# ) -> k2.Fsa:
#     """Get the decoding lattice from a decoding graph and neural
#     network output.
#     Args:
#       nnet_output:
#         It is the output of a neural model of shape `(N, T, C)`.
#       decoding_graph:
#         An Fsa, the decoding graph. It can be either an HLG
#         (see `compile_HLG.py`) or an H (see `k2.ctc_topo`).
#       supervision_segments:
#         A 2-D **CPU** tensor of dtype `torch.int32` with 3 columns.
#         Each row contains information for a supervision segment. Column 0
#         is the `sequence_index` indicating which sequence this segment
#         comes from; column 1 specifies the `start_frame` of this segment
#         within the sequence; column 2 contains the `duration` of this
#         segment.
#       search_beam:
#         Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
#         (less pruning). This is the default value; it may be modified by
#         `min_active_states` and `max_active_states`.
#       output_beam:
#          Beam to prune output, similar to lattice-beam in Kaldi.  Relative
#          to best path of output.
#       min_active_states:
#         Minimum number of FSA states that are allowed to be active on any given
#         frame for any given intersection/composition task. This is advisory,
#         in that it will try not to have fewer than this number active.
#         Set it to zero if there is no constraint.
#       max_active_states:
#         Maximum number of FSA states that are allowed to be active on any given
#         frame for any given intersection/composition task. This is advisory,
#         in that it will try not to exceed that but may not always succeed.
#         You can use a very large number if no constraint is needed.
#       subsampling_factor:
#         The subsampling factor of the model.
#     Returns:
#       An FsaVec containing the decoding result. It has axes [utt][state][arc].
#     """
#     dense_fsa_vec = k2.DenseFsaVec(
#         nnet_output,
#         supervision_segments,
#         allow_truncate=subsampling_factor - 1,
#     )

#     lattice = k2.intersect_dense_pruned(
#         decoding_graph,
#         dense_fsa_vec,
#         search_beam=search_beam,
#         output_beam=output_beam,
#         min_active_states=min_active_states,
#         max_active_states=max_active_states,
#     )

#     return lattice


# class MWERLoss(torch.nn.Module):
#     '''Minimum Word Error Rate Loss compuration in k2.

#     See equation 2 of https://arxiv.org/pdf/2106.02302.pdf about its definition.
#     '''

#     def __init__(
#         self,
#         vocab_size: int,
#         subsampling_factor: int,
#         search_beam: int = 20,
#         output_beam: int = 8,
#         min_active_states: int = 30,
#         max_active_states: int = 10000,
#         temperature: float = 1.0,
#         num_paths: int = 100,
#         use_double_scores: bool = True,
#         nbest_scale: float = 0.5,
#         reduction: Literal['none', 'mean', 'sum'] = 'mean'
#     ) -> None:
#         """
#         Args:
#           search_beam:
#             Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
#             (less pruning). This is the default value; it may be modified by
#             `min_active_states` and `max_active_states`.
#           output_beam:
#              Beam to prune output, similar to lattice-beam in Kaldi.  Relative
#              to best path of output.
#           min_active_states:
#             Minimum number of FSA states that are allowed to be active on any given
#             frame for any given intersection/composition task. This is advisory,
#             in that it will try not to have fewer than this number active.
#             Set it to zero if there is no constraint.
#           max_active_states:
#             Maximum number of FSA states that are allowed to be active on any given
#             frame for any given intersection/composition task. This is advisory,
#             in that it will try not to exceed that but may not always succeed.
#             You can use a very large number if no constraint is needed.
#           subsampling_factor:
#             The subsampling factor of the model.
#           temperature:
#             For long utterances, the dynamic range of scores will be too large
#             and the posteriors will be mostly 0 or 1.
#             To prevent this it might be a good idea to have an extra argument
#             that functions like a temperature.
#             We scale the logprobs by before doing the normalization.
#           use_double_scores:
#             True to use double precision floating point.
#             False to use single precision.
#           reduction:
#             Specifies the reduction to apply to the output:
#             'none' | 'sum' | 'mean'.
#             'none': no reduction will be applied.
#                     The returned 'loss' is a k2.RaggedTensor, with
#                     loss.tot_size(0) == batch_size.
#                     loss.tot_size(1) == total_num_paths_of_current_batch
#                     If you want the MWER loss for each utterance, just do:
#                     `loss_per_utt = loss.sum()`
#                     Then loss_per_utt.shape[0] should be batch_size.
#                     See more example usages in 'k2/python/tests/mwer_test.py'
#             'sum': sum loss of each path over the whole batch together.
#             'mean': divide above 'sum' by total num paths over the whole batch.
#           nbest_scale:
#             Scale `lattice.score` before passing it to :func:`k2.random_paths`.
#             A smaller value leads to more unique paths at the risk of being not
#             to sample the path with the best score.
#           num_paths:
#             Number of paths to **sample** from the lattice
#             using :func:`k2.random_paths`.
#         """
#         super().__init__()

#         self.vocab_size = vocab_size
#         self.search_beam = search_beam
#         self.output_beam = output_beam
#         self.min_active_states = min_active_states
#         self.max_active_states = max_active_states

#         self.num_paths = num_paths
#         self.nbest_scale = nbest_scale
#         self.subsampling_factor = subsampling_factor

#         self.mwer_loss = k2.MWERLoss(
#             temperature=temperature,
#             use_double_scores=use_double_scores,
#             reduction=reduction
#         )

#     def forward(
#         self,
#         emissions: torch.Tensor,
#         emissions_lengths: torch.Tensor,
#         labels: torch.Tensor,
#         labels_length: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Args:
#             emissions (torch.FloatTensor): CPU tensor of shape `(batch, frame, num_tokens)` storing sequences of
#                 probability distribution over labels; output of acoustic model.
#             labels (torch.FloatTensor): CPU tensor of shape `(batch, label_len)` storing labels.
#             emissions_lengths (Tensor or None, optional): CPU tensor of shape `(batch, )` storing the valid length of
#                 in time axis of the output Tensor in each batch.
#             labels_length (Tensor or None, optional): CPU tensor of shape `(batch, )` storing the valid length of
#                 label in each batch.

#         Returns:
#             torch.FloatTensor:
#                 Minimum Word Error Rate loss.
#         """

#         H = k2.ctc_topo(
#             max_token=self.vocab_size-1,
#             modified=False,
#             device=emissions.device,
#         )

#         supervision_segments = torch.stack(
#             (
#                 torch.tensor(range(emissions_lengths.shape[0])),
#                 torch.zeros(emissions_lengths.shape[0]),
#                 emissions_lengths.cpu(),
#             ),
#             1,
#         ).to(torch.int32)

#         lattice = get_lattice(
#             nnet_output=emissions,
#             decoding_graph=H,
#             supervision_segments=supervision_segments,
#             search_beam=self.search_beam,
#             output_beam=self.output_beam,
#             min_active_states=self.min_active_states,
#             max_active_states=self.max_active_states,
#             subsampling_factor=self.subsampling_factor,
#         )

#         token_ids = []
#         for i in range(labels_length.size(0)):
#             temp = labels[i, : labels_length[i]].cpu().tolist()
#             token_ids.append(list(filter(lambda num: num != 0, temp)))

#         loss = self.mwer_loss(
#             lattice, token_ids,
#             nbest_scale=self.nbest_scale,
#             num_paths=self.num_paths
#         )
#         print("mwer loss:", loss)

#         return loss