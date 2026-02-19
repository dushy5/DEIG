import torch
import torch.nn as nn
from functools import partial

from transformers import T5EncoderModel, T5Tokenizer

from ldm.modules.x_transformer import Encoder, TransformerWrapper


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt",
                                        return_offsets_mapping=True)
        tokens = batch_encoding["input_ids"].to(self.device)
        offset_mapping = batch_encoding["offset_mapping"]
        return tokens, offset_mapping

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text, return_offset_mapping=False):
        if self.use_tknz_fn:
            tokens, offset_mapping = self.tknz_fn(text)#.to(self.device)
        else:
            assert False 
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)

        if return_offset_mapping:
            return z, offset_mapping
        else:
            return z 

    def encode(self, text, return_offset_mapping=False):
        # output of length 77
        return self(text, return_offset_mapping)


def T5layerFFforward(self, hidden_states):
    with torch.cuda.amp.autocast(enabled=False): 
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
    return hidden_states

def replace_forward_method_for_T5(model):

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'T5LayerFF':
            from transformers.models.t5.modeling_t5 import T5LayerFF
            layer.forward = T5layerFFforward.__get__(layer, T5LayerFF)
        
        replace_forward_method_for_T5(layer)
    
    return model
        
class T5TextEmbedder(nn.Module):
    def __init__(self, pretrained_path="checkpoints/T5-XL", max_length=None, train=True):
        super().__init__()
        self.model = T5EncoderModel.from_pretrained(pretrained_path)
        if train:
            self.model = replace_forward_method_for_T5(self.model)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
        self.max_length = max_length

    def forward(
        self, caption, text_input_ids=None, attention_mask=None, max_length=None
    ):
        if max_length is None:
            max_length = self.max_length

        if text_input_ids is None or attention_mask is None:
            if max_length is not None:
                text_inputs = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
            else:
                text_inputs = self.tokenizer(
                    caption, return_tensors="pt", add_special_tokens=True
                )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
        text_input_ids = text_input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        outputs = self.model(text_input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state
        return embeddings
