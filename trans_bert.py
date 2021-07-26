import torch
import torch.nn as nn
from pytorch_transformers import *
from transformers_.modeling_bert import BertEmbeddings, BertPooler, BertLayer


class BertModel4Trans(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel4Trans, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Trans(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, trans_layer=-1, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, transform=None, train=True):

        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(torch.float32)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask,
            trans_layer=trans_layer, transform=transform, train=train)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class BertEncoder4Trans(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Trans, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states,  attention_mask=None, head_mask=None, transform=None,trans_layer=-1, train=True):
        all_hidden_states = ()
        all_attentions = ()

        if trans_layer == -1:
            hidden_states = transform(hidden_states)

        for i, layer_module in enumerate(self.layer):
            if i == trans_layer:
                if train:
                    hidden_states = transform(hidden_states)
                else:
                    pass
                    # print("Eval and not trans")

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class TransBert(nn.Module):
    def __init__(self, trans_option=False, transform=None):
        super(TransBert, self).__init__()

        self.trans_option = trans_option

        if trans_option:
            self.bert = BertModel4Trans.from_pretrained('bert-base-uncased')
            self.transform = transform
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')


    def forward(self, x, trans_layer=-1, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, train=True):
        if self.trans_option == True:
            all_hidden, pooler = self.bert(x, trans_layer, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, transform=self.transform, train=train)
        else: 
            all_hidden, pooler = self.bert(x, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,)

        return all_hidden, pooler 

