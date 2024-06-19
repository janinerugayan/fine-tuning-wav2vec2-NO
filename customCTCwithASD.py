import numpy as np
import torch
from dtw import *
import torch
import torch.nn.functional as F
from scipy.spatial import distance
import ctc_optimized  # cython built ctc loss & grad calc
import asd_for_ctc  # to extract ASD metric aligned to label seq for CTC loss calc
# from torchaudio.models.decoder import ctc_decoder  # for extracting nbest hypotheses
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_asd_score_batch(model, tokenizer, reference, hypothesis):
    asd_score = []
    for ref_text, hyp_text in zip(reference, hypothesis):
        ref_text = ref_text.replace("[UNK]", "")
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
    def forward(ctx, logits, seq, input_length):

        # ============= CYTHON CTC loss implementation =============
        params = logits.transpose(1,0)
        # convert logits to log probs
        params = params - (torch.max(params, dim=0)[0])
        params = torch.exp(params)
        params = params / torch.sum(params, dim=0)

        params_arr = params.double().detach().cpu().numpy()
        seq_arr = seq.int().detach().cpu().numpy()
        # cosdist_arr = np.array(cosdist_for_ctc, dtype=np.float64)
        llForward, llBackward, alphas, betas = ctc_optimized.forward_pass(params_arr, seq_arr, blank=31)

        alphas_tensor = torch.from_numpy(alphas).to(device)
        betas_tensor = torch.from_numpy(betas).to(device)
        llForward_tensor = torch.tensor(llForward).to(device)
        llBackward_tensor = torch.tensor(llBackward).to(device)
        # cosdist_tensor = torch.tensor(cosdist_for_ctc).to(device)

        ctx.save_for_backward(params, seq, input_length, alphas_tensor, betas_tensor, llForward_tensor,
                              llBackward_tensor)

        return llForward_tensor

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

        return (grad_tensor.transpose(1,0), None, None)


class MyCTC_nbest(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, seq, input_length, nbest_seq, lambda_asd):

        # ============= CYTHON CTC loss implementation =============
        params = logits.transpose(1,0)
        # convert logits to log probs
        params = params - (torch.max(params, dim=0)[0])
        params = torch.exp(params)
        params = params / torch.sum(params, dim=0)

        params_arr = params.double().detach().cpu().numpy()
        seq_arr = seq.int().detach().cpu().numpy()
        nbest_seq_arr = nbest_seq.int().detach().cpu().numpy()
        llForward, llBackward, alphas, betas = ctc_optimized.forward_pass_with_ASD_nbest(params_arr, seq_arr,
                                                                                         nbest_seq_arr,
                                                                                         lambda_asd, blank=31)

        alphas_tensor = torch.from_numpy(alphas).to(device)
        betas_tensor = torch.from_numpy(betas).to(device)
        llForward_tensor = torch.tensor(llForward).to(device)
        llBackward_tensor = torch.tensor(llBackward).to(device)

        ctx.save_for_backward(params, seq, input_length, alphas_tensor, betas_tensor, llForward_tensor,
                              llBackward_tensor)

        return llForward_tensor

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


# ===================================================
# USING STANF0RD-CTC CODE:
def compute_CTCloss_withASD(reference_text, predicted_text, ref_label_ids, output_logits, input_lengths, asd_model, asd_tokenizer, lambda_asd):
    loss = torch.zeros((len(reference_text)), requires_grad=True, device=device).double()
    for i in range(len(reference_text)):
        ref_text = reference_text[i].replace("[UNK]", "")
        pred_text = predicted_text[i].replace("[UNK]", "")
        label_ids = ref_label_ids[i]
        labels_mask = label_ids >= 0
        flattened_labels = label_ids.masked_select(labels_mask)
        logits = output_logits[i]
        # ref_alignments = asd_for_ctc.get_asd_align(ref_text, pred_text, asd_model, asd_tokenizer)
        # tokens_compressed = asd_for_ctc.get_per_token_cosdist(ref_alignments)
        # cosdist_for_ctc = asd_for_ctc.get_cosdist_for_ctc(tokens_compressed, flattened_labels)

        # max_per_frame = torch.argmax(logits, dim=1)
        # relevant_frames = []
        # for max_id in max_per_frame:
        #     if max_id in flattened_labels:
        #         relevant_frames.append(1)
        #     else:
        #         relevant_frames.append(0)

        myctcloss = MyCTC.apply
        # if len(flattened_labels) != len(cosdist_for_ctc):
        #     raise Exception("cosdist for ctc length not equal to flattened labels length")
        # loss[i] = myctcloss(logits, flattened_labels, input_lengths[i], cosdist_for_ctc, lambda_asd, relevant_frames)
        loss[i] = myctcloss(logits, flattened_labels, input_lengths[i])
    return loss.sum()


def compute_CTCloss_withASD_nbest(reference_text, predicted_text, ref_label_ids, output_logits, input_lengths, asd_model, asd_tokenizer, lambda_asd):

    # decoder = ctc_decoder(lexicon=None, tokens="tokens.txt", nbest=5, beam_size=50, blank_token="[PAD]",
    #                   sil_token="|", unk_word="[UNK]")

    loss = torch.zeros((len(reference_text)), requires_grad=True, device=device).double()

    for i in range(len(reference_text)):
        ref_text = reference_text[i].replace("[UNK]", "")
        pred_text = predicted_text[i].replace("[UNK]", "")
        label_ids = ref_label_ids[i]
        labels_mask = label_ids >= 0
        target_length = labels_mask.sum(-1)
        flattened_labels = label_ids.masked_select(labels_mask)
        logits = output_logits[i]

        # # extracting n-best hypotheses:
        # nbest_list = decoder(logits.detach().cpu()[None, :, :])
        # nbest_token_list = []
        # asd_score_list = [0] * len(nbest_list[0])
        # beam_score_list = [0] * len(nbest_list[0])
        # hyp_list = []
        # for i, item in enumerate(nbest_list[0]):
        #     tokens = item.tokens
        #     for i in range(len(tokens)):
        #         if tokens[i] == 0:
        #             tokens_mod = tokens[i+1:]
        #         else:
        #             break
        #     chars = decoder.idxs_to_tokens(tokens_mod)
        #     nbest_token_list.append(tokens_mod)
        #     hyp_text = re.sub(" +", " ", "".join(chars).replace("|", " "))
        #     hyp_list.append(hyp_text)
        #     asd_score_list[i] = compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text, hyp_text)
        #     beam_score_list[i] = item.score

        # # caculating the n-best approx of expected ASD score
        # asd_average = np.average(asd_score_list)
        # beam_score_sum = np.sum(beam_score_list)
        # asd_expected_list = [0] *  len(asd_score_list)
        # for i in range(len(asd_expected_list)):
        #     asd_expected_list[i] = (beam_score_list[i] / beam_score_sum) * (asd_score_list[i] - asd_average)
        # asd_expected_sum = np.sum(asd_expected_list)
        # asd_expected_tensor = torch.tensor(asd_expected_sum, requires_grad=True, device=device)

        # nbest_token_seq = nbest_token_list[np.argmin(asd_score_list)]
        # best_asd_score = asd_score_list[np.argmin(asd_score_list)]

        # if best_asd_score < compute_asd_score_single_utt(asd_model, asd_tokenizer, ref_text, pred_text):
        #     print("predicted:", pred_text)
        #     print("nbest-1st:", hyp_list[np.argmin(asd_score_list)])

        # log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        # print(log_probs.shape)
        # with torch.backends.cudnn.flags(enabled=False):
        #     orig_loss = F.ctc_loss(
        #         log_probs,
        #         flattened_labels,
        #         input_lengths[i],
        #         target_length,
        #         blank=31,
        #         reduction="mean",
        #         zero_infinity=True,
        #     )
        #     nbest_loss = F.ctc_loss(
        #         log_probs,
        #         nbest_token_seq,
        #         input_lengths[i],
        #         torch.tensor(len(nbest_token_seq)),
        #         blank=31,
        #         reduction="mean",
        #         zero_infinity=True,
        #     )
        # loss[i] = ((1 - lambda_asd) * orig_loss) + (lambda_asd * asd_expected_tensor)
        # print(loss[i])

        myctcloss = MyCTC.apply
        # loss[i] = ((1 - lambda_asd) * myctcloss(logits, flattened_labels, input_lengths[i])) + (lambda_asd * asd_expected_tensor)
        loss[i] = myctcloss(logits, flattened_labels, input_lengths[i])
        print(loss[i])

    return loss.sum()


# ===================================================
# CTC with Gumbel-Softmax sampled ASD scoring
def sampled_logits_asd_loss(reference_text, predicted_text, output_logits, metric_model, metric_tokenizer, processor):
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