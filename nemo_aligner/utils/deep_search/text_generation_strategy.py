# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import copy
import os
import re
import warnings
from typing import List, Set, Tuple

import torch
from transformers import CLIPImageProcessor

from nemo.collections.nlp.modules.common.lm_utils import pad_batch
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo_aligner.utils.deep_search.forward_only import get_forward_output_only_func
from megatron.core import InferenceParams, parallel_state
from nemo.collections.nlp.modules.common.megatron.utils import build_attention_mask_3d


try:
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches
    from apex.transformer.enums import AttnMaskType
    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


# the text representation of eos_id, it applies for all tokenizers
END_OF_SEQ = '<|endoftext|>'


def front_pad_batch(batch, pad_id, max_len, eos_id):
    context_lengths = []
    max_context_length = max([len(tokens) for tokens in batch])
    new_batch = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < max_context_length:
            new_batch.append([eos_id] * (max_context_length - context_length) + tokens + [pad_id] * max_len)
        else:
            new_batch.append(tokens + [pad_id] * max_len)
        context_lengths.append(max_context_length)
    return new_batch, context_lengths


class TextGenerationStrategy:
    """
    Base class for TextGeneration Strategy
    """

    def __init__(self, model):
        self.model = model
        if self.model.training:
            # TODO in the future this should raise an exception
            warnings.warn(
                "Generation started while the model is in training mode, switching to eval mode "
                "(this situation may raise an exception in future versions, please call `eval()` before generation)"
            )
            self.model.eval()
        self._end_of_generation_cache = None

    def forward_step(self, batch, tensor_shape):
        func = get_forward_output_only_func(self)

        fwd_bwd_function = get_forward_backward_func()
        output_tensor = fwd_bwd_function(
            forward_step_func=func,
            data_iterator=iter([batch,]),
            model=[self.forward_model],
            num_microbatches=get_num_microbatches(),
            forward_only=True,
            seq_length=tensor_shape[0],
            micro_batch_size=tensor_shape[1],
        )

        return output_tensor

    def tokenize_batch(self, sentences, max_len, add_BOS):
        """
        convert the sentences into lists of tokens, pad them to the same length, add bos tokens if it is needed
        Args:
            sentences (List[str]): list of input sentences in str format.
            max_len (int): max number of tokens to generate.
            add_BOS (bool): whether to add the BOS token at the beginning
        Returns:
            Tuple[torch.Tensor], the tokenized and padded torch tensor and the token context length tensor.
        """
        tokenizer = self.model.tokenizer
        if add_BOS:
            context_tokens = [[tokenizer.bos_id] + tokenizer.text_to_ids(s) for s in sentences]
        else:
            context_tokens = [tokenizer.text_to_ids(s) for s in sentences]
        context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id, max_len)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

    @abc.abstractclassmethod
    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length
        Args:
            maxlen (int): the max len computed from the context and number of tokens to generate
        returns (int):
            the clip the max length based of the LM model max sequence length
        """
        pass

    @abc.abstractclassmethod
    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
        """initialize the batch data before the inference steps.
           It will save the intermediate results as object attributes
           context_length (int): the context token length
           compute_attention_mask: bool: set to True to compute attention mask (not needed for FA)
        Args:
            context_tokens (torch.Tensor):  The padded context tokens including the space for tokens to be generated 
        """
        pass

    @abc.abstractclassmethod
    def prepare_batch_at_step(
        self, tokens: torch.Tensor, maxlen: int, micro_batch_size: int, step: int, context_length: int
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        generate the batch used in inference for each of the steps
        Args:
            tokens  (torch.Tensor): the context tokens
            maxlen (int): the maximum length in the context tokens
            micro_batch_size (int): text generation batch size
            step (int): the inference step count
            context_length (int): the new token position in the tokens
        returns:
            a tuple of list of tensor arguments for the model and a list of tensor shape required by forward method
        """
        pass

    @abc.abstractclassmethod
    def post_process(self, tokens: torch.Tensor, new_tokens: torch.Tensor, context_length: int):
        """
        At the end of the single step inference, post process the inference results
        Args:
            tokens  (torch.Tensor): the context tokens
            new_token (torch.Tensor): sampled new token id
            context_length (int): the new token position in the tokens
        """
        pass

    def end_of_generation_condition(
        self, tokens: torch.Tensor, prev: torch.Tensor, eod_id: int, end_strings: List[str]
    ) -> torch.Tensor:
        """
        return whether the generation should stop based on the previous token
        Args:
            tokens (torch.Tensor): the generated tokens so far
            prev  (torch.Tensor): the previous token
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        returns:
            a boolean tensor indicating whether the generation should stop
        """
        if (len(end_strings) == 1 and end_strings[0] == END_OF_SEQ) or not end_strings:
            # Simple scenario: only finish on end of document token.
            return prev == eod_id

        end_tokens, end_strings_to_check = self._get_end_of_generation_tokens_and_strings(eod_id, end_strings)
        assert end_tokens

        is_end = torch.isin(prev, torch.tensor(list(end_tokens), dtype=prev.dtype, device=prev.device))

        if end_strings_to_check:
            # The loop below is inefficient (see warning in `_get_end_of_generation_tokens_and_strings()`)
            # TODO In addition, we will not stop if the model generates an end string followed by extra characters,
            # e.g., if `end_string` is "Done" and there exists a "Done!" token it could generate tokens
            #       [..., ".", "Done!"]
            # which would fail the `endswith("Done")` check. However, stopping when "Done!" is generated would not
            # work either, since we would need to post-process the generated string to truncate the extra "!".
            # ==> this is left for future work if there is a compelling use case requiring this feature.
            for idx, token_seq in enumerate(tokens):
                text = self.model.tokenizer.ids_to_text(token_seq.tolist())
                is_end[idx] |= any(text.endswith(end_string) for end_string in end_strings_to_check)

        return is_end

    def post_generation_process(self, output):
        """
        At the end of the text generation, post process the results
        Args:
            output  (dict): the text generation output dictionary
        """
        return output

    def _get_end_of_generation_tokens_and_strings(
        self, eod_id: int, end_strings: List[str]
    ) -> Tuple[Set[int], List[str]]:
        """
        return the tokens and strings indicating the end of generation
        Args:
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        Returns:
            a pair `(tokens, strings)` where `tokens` is a set of tokens (int) and `strings` is a list of strings,
            which must all be used to identify the end of generation (`tokens` always contains `eod_id`, while
            `strings` may be empty if all end strings are associated to unique tokens)
        """
        tokenizer = self.model.tokenizer
        # A cache is used to remember which end strings are associated to unique tokens vs. which ones
        # require an actual string comparison.
        if self._end_of_generation_cache is None or self._end_of_generation_cache["tokenizer"] is not tokenizer:
            # Invalidate the cache.
            self._end_of_generation_cache = {
                "tokenizer": tokenizer,
                "end_string_to_token": {END_OF_SEQ: eod_id},
                "end_strings_to_check": set(),
            }
        end_string_to_token = self._end_of_generation_cache["end_string_to_token"]

        end_tokens = {eod_id}  # always include `eod_id`, even if `END_OF_SEQ` is not within `end_strings`
        end_strings_to_check = []  # will contain end strings that have no associated special token

        for end_string in end_strings:
            try:
                end_tokens.add(end_string_to_token[end_string])
                continue
            except KeyError:
                if end_string in self._end_of_generation_cache["end_strings_to_check"]:
                    end_strings_to_check.append(end_string)
                    continue

            # `end_string` does not exist in the cache yet: check if `end_string` is a special token for
            # the tokenizer. Ideally, we would simply use `tokenizer.text_to_ids(end_string)`, but some
            # tokenizers (e.g., SentencePiece) may prefix the special token with another token associated
            # to an empty string. The code below is thus meant to extract the special token associated to
            # `end_string` (if it exists). Note that we use "<extra_id_1>" as prefix string to have a low
            # risk of the tokenizer merging it with `end_string`, but this is somewhat arbitrary.
            ids_ref = tokenizer.text_to_ids("<extra_id_1>")
            ids_with_end_string = tokenizer.text_to_ids(f"<extra_id_1>{end_string}")
            if len(ids_with_end_string) == len(ids_ref) + 1 and ids_with_end_string[:-1] == ids_ref:
                # We can assume that the extra token is the one corresponding to `end_string`.
                end_string_to_token[end_string] = ids_with_end_string[-1]
                end_tokens.add(ids_with_end_string[-1])
            else:
                # No special token.
                warnings.warn(
                    f"The end string '{end_string}' has no associated special token: this may slow down "
                    "generation (consider using a different tokenizer or modifying `end_strings`)"
                )
                self._end_of_generation_cache["end_strings_to_check"].add(end_string)
                end_strings_to_check.append(end_string)

        return end_tokens, end_strings_to_check


class GPTModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model):
        super().__init__(model)
        self.forward_model = self.model.model
        # all the model parallel worker will have a copy of the inference params
        # to store the K-V cache
        self.inference_params = None

    def tokenize_batch(self, sentences, max_len, add_BOS):
        """
        convert the sentences into lists of tokens, pad them to the same length, add bos tokens if it is needed
        Args:
            sentences (List[str]): list of input sentences in str format.
            max_len (int): max number of tokens to generate.
            add_BOS (bool): whether to add the BOS token at the beginning
        Returns:
            Tuple[torch.Tensor], the tokenized and padded torch tensor and the token context length tensor.
        """
        tokenizer = self.model.tokenizer
        if add_BOS:
            context_tokens = [[tokenizer.bos_id] + tokenizer.text_to_ids(s) for s in sentences]
        else:
            context_tokens = [tokenizer.text_to_ids(s) for s in sentences]
        context_tokens, context_lengths = front_pad_batch(context_tokens, -1, max_len, -2)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length"""

        # for positional embedding types that allow length extrapolation, don't clip the max length
        if self.model.cfg.get("position_embedding_type", "learned_absolute") == "learned_absolute":
            if maxlen > self.model.cfg.encoder_seq_length + 1:
                maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        tokenizer = self.model.tokenizer
        tokens = context_tokens.contiguous().cuda()
        
        seq_length = tokens.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
        self.position_ids = position_ids.unsqueeze(0).repeat(tokens.size(0), 1)

        token_mask = tokens != -2
        front_pad_lengths = torch.sum(tokens == -2, dim=1)
        tokens[tokens == -1] = tokenizer.eos_id
        tokens[tokens == -2] = tokenizer.eos_id
        self.attention_mask = build_attention_mask_3d(token_mask, token_mask, attn_mask_type=AttnMaskType.causal)

        for b in range(tokens.size(0)):
            offset = front_pad_lengths[b]
            self.position_ids[b, offset:] -= offset

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_length: int,
        compute_attention_mask: bool = True,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        generate the batch used in inference for each of the steps
        """
        # types2use = None
        if step == 0:
            # Allocate memory for the entire context.
            tokens2use = tokens[:, :context_length]
            positions2use = self.position_ids[:, :context_length]
            # init inference params 
            self.inference_params = InferenceParams(
                max_batch_size=micro_batch_size, max_sequence_length=maxlen
            )
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, :context_length]
        else:
            if step == 1:
                self.inference_params.sequence_len_offset = context_length - 1
            else:
                self.inference_params.sequence_len_offset += 1
            # Set this to false so the memory is not reallocated.
            tokens2use = tokens[:, context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, context_length - 1].view(micro_batch_size, -1)
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, context_length - 1].view(batch_size, -1)

        """Prepare batch for each of the inference steps"""
        attention_mask_repeat = None
        if compute_attention_mask:
            attention_mask_repeat = torch.concat([self.attention_mask for _ in range(micro_batch_size)])

        batch = [tokens2use, attention_mask_repeat, positions2use]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape