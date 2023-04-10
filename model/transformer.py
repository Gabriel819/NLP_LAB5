import torch
import torch.nn as nn
from model.utils import pad_mask, masked_attn_mask
from model.sub_layers import Embedding_Layer, Encoder, Decoder

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    def __init__(self, num_token, max_seq_len, dim_model, d_k=64, d_v=64, n_head=8, dim_hidden=2048, d_prob=0.1, n_enc_layer=6, n_dec_layer=6):
        super(Transformer, self).__init__()

        """
        each variable is one of example, so you can change it, it's up to your coding style.
        """
        self.num_token = num_token
        self.max_seq_len = max_seq_len
        self.input_embed = nn.Embedding(num_token, dim_model)
        self.output_embed = nn.Embedding(num_token, dim_model)
        self.enc_embed = Embedding_Layer(num_token=num_token, dim_model=dim_model, max_seq_len=max_seq_len, d_prob=d_prob) # num_token: 54887, dim_model: 512, max_seq_len: 20, d_prob: 0.2(dropout probability)
        self.dec_embed = Embedding_Layer(num_token=num_token, dim_model=dim_model, max_seq_len=max_seq_len, d_prob=d_prob)  # num_token: 54887, dim_model: 512, max_seq_len: 20, d_prob: 0.2(dropout probability)
        self.encoder = Encoder(dim_model, d_k, d_v, n_head, dim_hidden, d_prob, n_enc_layer)
        self.decoder = Decoder(dim_model, d_k, d_v, n_head, dim_hidden, d_prob, n_dec_layer)
        self.linear = nn.Linear(dim_model, num_token)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, tgt, teacher_forcing=True):
        """
        ToDo: feed the input to Encoder and Decoder
        """
        # tgt - start: 0 token[SOS], end: 1 token[EOS]
        # src: (128, 20), tgt: (128, 21)
        # 0. Make PAD Mask from the src
        padding_mask = pad_mask(src)

        # 1. Positional embedding for Encoder
        src_embed = self.input_embed(src)  # (128, 20, 512)
        pos_embed = self.enc_embed(src).to(device)
        enc_input = src_embed + pos_embed

        # 2. Encoder
        enc_out = self.encoder(enc_input, padding_mask) # enc_out: (128, 20, 512)

        # 4. Positional embedding for Decoder
        tgt_embed = self.output_embed(tgt) # (128, 21, 512)
        tgt_pos_embed = self.dec_embed(tgt).to(device) # (128, 21, 512)
        dec_input = tgt_embed + tgt_pos_embed  # (128, 21, 512)

        # 5. Decoder
        dec_pad_mask = pad_mask(tgt[:,:-1])
        decoder_out = self.decoder(enc_out, dec_input[:,:-1,:], dec_pad_mask)

        # 6. Linear layer
        out = self.linear(decoder_out)

        # 7. Softmax
        out = self.softmax(out)
        b, l, t = out.shape

        # 8. add sos token
        res = torch.zeros(b, l+1, t)
        res[:, 1:, :] = out

        return res

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
