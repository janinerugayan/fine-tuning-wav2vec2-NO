import torch
from transformers.models.wav2vec2.modeling_wav2vec2 import *
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoTokenizer, BertModel, Wav2Vec2ProcessorWithLM, Wav2Vec2Processor
from torch import nn
import torch.nn.functional as F


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
    for label in label_ids:
        if label == 0 and len(cosdist_for_ctc) == 0:
            cosdist_for_ctc.append(0)
        elif label != 0:
            cosdist_for_ctc.append(tokens_compressed[token_count][2])
        elif label == 0:
            token_count += 1
            cosdist_for_ctc.append(0)
    if len(cosdist_for_ctc) != len(label_ids):
        print("mismatch in number of tokens compressed and tokens identified from label ids")
        print("cosdist: ", len(cosdist_for_ctc), "label_ids: ", len(label_ids))
        return cosdist_for_ctc
    else:
        return cosdist_for_ctc


# INCORPORATING ASD COSDIST VALUES TO THE CTC CALCULATION
def calculate_CTC_with_ASD(params, seq, cosdist_for_ctc, blank=0):
    seqLen = seq.shape[0]  # length of label sequence
    L = 2*seqLen + 1  # length of the label sequence with blanks
    T = params.shape[1]  # length of utterance (time)

    alphas = torch.zeros((L,T)).double()
    betas = torch.zeros((L,T)).double()

    # convert logits to log probs
    params = params - (torch.max(params, dim=0)[0])
    params = torch.exp(params)
    params = params / torch.sum(params, dim=0)

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
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t] * (1 - cosdist_for_ctc[l])
            else:
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) * params[seq[l],t] * (1 - cosdist_for_ctc[l])

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
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t] * (1 - cosdist_for_ctc[l])
            else:
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) * params[seq[l],t] * (1 - cosdist_for_ctc[l])

        # normalize at current time
        c = torch.sum(betas[start:end,t])
        betas[start:end,t] = betas[start:end,t] / c
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
        return (-llForward, grad)
    else:
        grad = params - grad / (params * absum)
        return (-llForward, grad)


def ctc_loss_with_ASD(batch_logits, batch_labels, asd_model, asd_tokenizer, processor, processor_woLM, blank=0):
    ctc_loss_with_asd = 0
    reference_text = processor_woLM.batch_decode(batch_labels, group_tokens=False)
    predicted_text = processor.batch_decode(batch_logits.detach())
    for i in range(batch_logits.shape[0]):
        labels = batch_labels[i]
        ref = reference_text[i].replace("[UNK]", "")
        hyp = predicted_text[i].replace("[UNK]", "")
        print(i, ref)
        print(i, hyp)
        ref_alignments = get_asd_align(ref, hyp, asd_model, asd_tokenizer)
        tokens_compressed = get_per_token_cosdist(ref_alignments)
        cosdist_for_ctc = get_cosdist_for_ctc(tokens_compressed, labels)
        loss, __ = ctc_loss_with_ASD(logits.transpose(1,0), labels, cosdist_for_ctc, blank=0)
        ctc_loss_with_asd += loss

    ctc_loss_with_asd = ctc_loss_with_asd / batch_logits.shape[0]

    return ctc_loss_with_asd


class Wav2Vec2ForCTCwithASD(Wav2Vec2PreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `withASD.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # for CTC with ASD incorporated
        self.asd_model = BertModel.from_pretrained('ltg/norbert2')
        self.asd_tokenizer = AutoTokenizer.from_pretrained('ltg/norbert2')
        self.processor = Wav2Vec2ProcessorWithLM.from_pretrained("NbAiLab/nb-wav2vec2-300m-bokmaal")
        self.processor_woLM = Wav2Vec2Processor.from_pretrained("NbAiLab/nb-wav2vec2-300m-bokmaal")

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """

        # Note that `tie_weights` is usually used to tie input and output embedding weights. The method is re-purposed to
        # correctly load adapter layers for Wav2Vec2 so that we do not have to introduce a new API to
        # [`PreTrainedModel`]. While slightly hacky, Wav2Vec2 never has to tie input and output embeddings, so that it is
        # ok to repurpose this function here.
        target_lang = self.target_lang

        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            print(logits.shape)
            print(flattened_targets.shape)

            # ctc loss with ASD incorporated
            loss = ctc_loss_with_ASD(
                logits,
                flattened_targets,
                self.asd_model,
                self.asd_tokenizer,
                self.processor,
                self.processor_woLM,
                blank=0
            )

            # ctc_loss doesn't support fp16
            # log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # with torch.backends.cudnn.flags(enabled=False):
            #     loss = nn.functional.ctc_loss(
            #         log_probs,
            #         flattened_targets,
            #         input_lengths,
            #         target_lengths,
            #         blank=self.config.pad_token_id,
            #         reduction=self.config.ctc_loss_reduction,
            #         zero_infinity=self.config.ctc_zero_infinity,
            #     )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )