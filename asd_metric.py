import datasets
from dtw import *
import torch

_DESCRIPTION = """\
Aligned Semantic Distance (ASD)

Extracts the embedding vector for the reference and hypothesis text using a transformer language model.

ASD value is the accumulated distance resulting from the optimal alignment from the two embedding vectors, determined using dynamic programming.

Cosine distance is the local distance metric used to compare the token embeddings with each other.
"""

_KWARGS_DESCRIPTION = """
Compute ASD score of hypothesis and reference text.

Args:
    references:
    hypothesis:

Returns:
    (float): the ASD score
"""

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ASD(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION
            inputs_description=_KWARGS_DESCRIPTION
        )

    def _compute(self, model, tokenizer, reference=None, hypothesis=None):
        tokenized_ref = tokenizer(reference, padding=True, truncation=True, max_length=512, return_tensors="pt")
        tokenized_hyp = tokenizer(hypothesis, padding=True, truncation=True, max_length=512, return_tensors="pt")
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
        min_global_distance_norm = (alignment.distance / num_tokens)
        return min_global_distance_norm
