import math
import os
import warnings
from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import BertForSequenceClassification, BertModel
from transformers.models.bert.modeling_bert import (
    BertSelfOutput,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
    ACT2FN,
    apply_chunking_to_forward,
    BERT_INPUTS_DOCSTRING,
    BertAttention,
    BertIntermediate,
    BertOutput
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    SequenceClassifierOutput
)
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

import copy

logger = logging.get_logger(__name__)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None, group_prob=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    num_attention_heads = query.size(-3)
    device = value.device
    # print("query", query.size())

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        seq_len = query.size()[-2]
        b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32), 0)).to(device)
        combined_mask = []
        for row in mask:
            combined_mask.append(row | b)
        combined_mask = torch.stack(combined_mask)
        # print("combined", combined_mask)
        # NOTE: this may not be correct
        combined_mask = combined_mask.unsqueeze(-3).repeat(1, num_attention_heads, 1, 1)
        # print("combined unsqueezed", combined_mask)
        # print("scores", scores.size())
        scores = scores.masked_fill(combined_mask == 0, -1e9)
    if group_prob is not None:
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = p_attn * group_prob.unsqueeze(1)
    else:
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_attention_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_attention_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_attention_heads
        self.h = num_attention_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, group_prob=None, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query,key,value shape: (nbatches, h, seq_len, d_k)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout, group_prob=group_prob)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class GroupAttention(nn.Module):
    """
    Computes neighboring attention, which gives a probability that the token should link to the previous and
    following token.
    """
    def __init__(self, config):
        super(GroupAttention, self).__init__()
        self.d_model = config.hidden_size

        self.linear_key = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.IntTensor],
                prior: torch.FloatTensor):
        """
        Computes constituent attention, a sequence a_1...a_n of probabilities where a_i is the probability that token
        w_i and neighbor word w_i+1 are the same constituent. Also computes the *constituent prior* C, a matrix (i, j)
        where C_i,j is the probability that w_i and w_j are in the same constituent.

        :param hidden_states: Token hidden states at a layer
        :param attention_mask: Attention mask for tokens
        :param prior: Group attention weights from the previous layer
        :return: (C, a)
        """
        device = hidden_states.device

        batch_size, seq_len = hidden_states.size()[:2]

        hidden_states = self.norm(hidden_states)

        # Selects, for each item in the sequence, the item before (a), the item (b), and the item after (c)
        a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), 1)).to(device)
        b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32), 0)).to(device)
        c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), -1)).to(device)
        tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len, seq_len], dtype=np.float32), 0)).to(device)

        # Mask for just the previous and next token
        mask = []
        for row in attention_mask:
            mask.append(row & (a+c))
        mask = torch.stack(mask)
        # mask = attention_mask & (a + c)

        key = self.linear_key(hidden_states)
        query = self.linear_query(hidden_states)

        # Compute attention scores between all states
        scores: torch.Tensor = torch.matmul(query, key.transpose(-2, -1)) / self.d_model

        # Mask out scores other than previous & next
        scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax so token must attend to either previous or next, but not both
        neighboring_attention = torch.nn.functional.softmax(scores, dim=-1)
        neighboring_attention = torch.sqrt(neighboring_attention * neighboring_attention.transpose(-2, -1) + 1e-9)

        neighboring_attention = prior + (1. - prior) * neighboring_attention

        t = torch.log(neighboring_attention + 1e-9).masked_fill(a == 0, 0).matmul(tri_matrix)
        constituent_attention = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
        constituent_attention = constituent_attention + constituent_attention.transpose(-2, -1) \
                                + neighboring_attention.masked_fill(b == 0, 1e-9)

        return constituent_attention, neighboring_attention


class TreeBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MultiHeadedAttention(num_attention_heads=config.num_attention_heads,
                                              d_model=config.hidden_size,
                                              dropout=config.attention_probs_dropout_prob)

        self.group_attention = GroupAttention(config)
        if config.is_decoder:
            raise Exception("Cannot initialize TreeBertLayer in decoder mode, unsupported")

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        constituent_prior,
        attention_mask: Optional[torch.IntTensor] = None,
        extended_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # Compute constituency attention in order to mask attention
        constituent_attention_output, neighboring_attention = self.group_attention(hidden_states, attention_mask, constituent_prior)

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_output: torch.Tensor = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            group_prob=constituent_attention_output,
            mask=attention_mask,
        )

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, (self_attention_output,)
        )
        outputs = (layer_output, constituent_attention_output, neighboring_attention, self_attention_output)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output[0])
        layer_output = self.output(intermediate_output, attention_output[0])
        return layer_output


class BaseModelOutputWithPastAndCrossAttentionsAndConstituentAttention(BaseModelOutputWithPastAndCrossAttentions):
    break_probs: Optional[Tuple[torch.FloatTensor]] = None


"""
Like a BERT encoder, but adds a *constituent prior*, which constrains items to only attend to 
other items in the same constituent at each level.
"""
class TreeBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([TreeBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        extended_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentionsAndConstituentAttention]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None

        # Track breakpoints so we can recover constituency if desired
        break_probs = []

        # Keep previous layer constituent attentions
        constituent_prior = 0.

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    constituent_prior,
                    attention_mask,
                    extended_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    constituent_prior,
                    attention_mask,
                    extended_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[3],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

            constituent_prior = layer_outputs[1]
            break_probs.append(layer_outputs[2])

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        output = BaseModelOutputWithPastAndCrossAttentionsAndConstituentAttention(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        output.break_probs = tuple(break_probs)
        return output



class BaseModelOutputWithPoolingAndCrossAttentionsAndConstituentAttention(BaseModelOutputWithPoolingAndCrossAttentions):
    break_probs: Optional[Tuple[torch.FloatTensor]] = None


class TreeBertModel(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        # Replace encoder block with Tree Encoder
        self.encoder = TreeBertEncoder(config)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentionsAndConstituentAttention]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            extended_attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        output = BaseModelOutputWithPoolingAndCrossAttentionsAndConstituentAttention(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        output.break_probs = encoder_outputs.break_probs
        return output


class SequenceClassifierOutputWithConstituentAttention(SequenceClassifierOutput):
    break_probs: Optional[Tuple[torch.FloatTensor]] = None

@add_start_docstrings(
    """
    Tree Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """
)
class TreeBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

        self.bert = TreeBertModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithConstituentAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: BaseModelOutputWithPoolingAndCrossAttentionsAndConstituentAttention = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        seq_outputs = SequenceClassifierOutputWithConstituentAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        seq_outputs.break_probs = outputs.break_probs
        # print(outputs.break_probs)
        return seq_outputs
