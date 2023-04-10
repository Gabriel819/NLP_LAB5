import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model.utils import masked_attn_mask
"""
Todo: Code Transformer sub-Layers

Todo: Please left comment when you code each module (You don't need to explain details)
i.e.
# Multi-head attention
def Multi_Head_Attn(...)
    ...
    return
"""
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class Embedding_Layer(nn.Module):
    def __init__(self, num_token, dim_model, max_seq_len, d_prob):
        super().__init__()
        # self.input_embed = nn.Embedding(num_token, dim_model)
        self.num_token = num_token
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.d_prob = d_prob

    def forward(self, src):
        # src: (128, 20), e.g. tensor([198, 35149, 5308, 8, 35150, 4, 78, 5308, 15, 12, 35151, 13, 5, 2, 2, 2, 2, 2, 2,2], device='cuda:0')
        b, l = src.shape
        # src_embed = self.input_embed(src) # (128, 20, 512)

        res_pos_embed = torch.zeros(l, self.dim_model, requires_grad=False) # (20, 512). Positional Embedding shouldn't be back-propagated.

        for pos in range(l): # max_length: 20
            for i in range(self.dim_model): # dim_model: 512
                # angle = 1/math.pow(10000, (2*(i//2))/self.dim_model)
                angle = pos / math.pow(10000, (2*i)/self.dim_model)
                if i % 2==0: # if even number,
                    res_pos_embed[pos, i] = math.sin(angle) # use sine function
                else: # if odd number,
                    res_pos_embed[pos, i] = math.cos(angle) # use cosine function

        # visualize and check the positional encoding result
        # plt.pcolormesh(res_pos_embed.numpy(), cmap='RdBu')
        # plt.xlabel('Depth')
        # plt.xlim((0, 128))
        # plt.ylabel('Position')
        # plt.colorbar()
        # plt.show()

        return res_pos_embed.squeeze(0).repeat(b, 1, 1) # add pos_embed into the input

class Encoder(nn.Module):
    def __init__(self, dim_model, d_k, d_v, n_head, dim_hidden, d_prob, n_enc_layer):
        super().__init__()
        self.dim_model = dim_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.dim_hidden = dim_hidden
        self.d_prob = d_prob
        self.n_enc_layer = n_enc_layer
        self.dropout_layer = nn.Dropout(p=d_prob)

        self.MultiHeadAttentionLayer = MultiHeadAttention(dim_model, dim_hidden, n_head, d_k, d_v)
        self.FeedForwardLayer = FeedForward(dim_model, dim_hidden)
        self.layernorm_layer = nn.LayerNorm(dim_model)

    def forward(self, src, padding_mask):
        b, l, d_m = src.shape # (128, 20, 512)
        out = src

        for idx in range(self.n_enc_layer):
            inp = out # (128, 20, 512) *** (2560,512) X
            # Multi-head-attention layer
            out = self.MultiHeadAttentionLayer(out, padding_mask) # (128, 20, 512) *** (2560, 512)
            # Dropout
            out = self.dropout_layer(out) # (128, 20, 512)

            # Add
            out = inp + out # (128, 20, 512)
            # Norm
            out = self.layernorm_layer(out) # (128, 20, 512)

            inp = out
            # Feed Forward Layer
            out = self.FeedForwardLayer(out)
            # Dropout
            out = self.dropout_layer(out)

            # Add
            out = inp + out
            # Norm
            out = self.layernorm_layer(out) # (128, 20, 512)

        return out

class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_hidden):
        super().__init__()
        self.dim_model = dim_model
        self.dim_hidden = dim_hidden
        # self.w_1 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(dim_model, dim_hidden)))
        # self.w_2 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(dim_hidden, dim_model)))
        # self.b_1 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(1,dim_hidden)))
        # self.b_2 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(1,dim_model)))
        self.fc_1 = nn.Linear(dim_model, dim_hidden)
        self.fc_2 = nn.Linear(dim_hidden, dim_model)

    def forward(self, x):
        out = self.fc_1(x) # (2560, 2048)
        out = self.fc_2(out) # (2560, 512)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, dim_hidden, n_head, d_k, d_v):
        super().__init__()
        self.dim_model = dim_model
        self.dim_hidden = dim_hidden
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_o = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_head*d_k, dim_model))) # w_o: (512, 512)

    def forward(self, src, padding_mask, dec_inp=None): # get query, key, value as inputs. (128, 20, 64)
        out_list = []
        # src = src.flatten(0, 1) # (2560, 64)

        for idx in range(self.n_head):
            attention_layer = Attention(self.dim_model, self.dim_hidden, self.d_k, self.d_v)
            out = attention_layer(src, padding_mask, dec_input=dec_inp) # (128, 20, 64)
            out_list.append(out)

        out_cat = torch.cat(out_list, dim=2) # out_cat: (128, 20, 512)
        res = torch.matmul(out_cat, self.w_o) # res: (128, 20, 512)

        return res

class Attention(nn.Module):
    def __init__(self, dim_model, dim_hidden, d_k, d_v):
        super().__init__()
        self.dim_model = dim_model
        self.dim_hidden = dim_hidden
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(dim_model, d_k))).to(device) # (512, 64)
        self.w_k = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(dim_model, d_k))).to(device) # (512, 64)
        self.w_v = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(dim_model, d_v))).to(device) # (512, 64)

        self.softmax_layer = nn.Softmax(dim=1)  # dim=1이 맞긴 한가?

    def forward(self, src, padding_mask, dec_input=None, sub_sequence_mask=None):
        b, l, d_m = src.shape  # (128, 20, 512)
        # src: (128, 20, 512), padding_mask: (128, 20)

        Q = torch.bmm(src, self.w_q.repeat(b,1,1)) # (128, 20, 64)
        K = torch.bmm(src, self.w_k.repeat(b,1,1)) # (128, 20, 64)
        if dec_input == None:
            V = torch.bmm(src, self.w_v.repeat(b,1,1)) # (128, 20, 64)
        else:
            V = torch.matmul(dec_input, self.w_v.repeat(b,1,1))

        # 1. MatMul Query and Key
        tmp = torch.bmm(Q, K.transpose(1,2))  # tmp: (128, 20, 20)
        # 2. Scale
        tmp = tmp / math.sqrt(self.d_k)  # /8 -> tmp: (128, 20, 20)
        # 3. PAD Mask & Sub-Sequence Mask(only for Masked Multi-Head Attention Layer)
        if sub_sequence_mask != None: # when Masked Multi-Head Layer
            # Pad mask
            tmp_pad_mask = padding_mask.unsqueeze(2).repeat(1, 1, l)  # tmp_pad_mask: (128, 20, 20)
            tmp = tmp * tmp_pad_mask  # row-wise padding
            tmp = tmp * tmp_pad_mask.transpose(1, 2)  # column-wise padding
            # Sub-sequence Mask
            tmp = tmp * sub_sequence_mask.to(device)
        # 4. Softmax
        tmp = self.softmax_layer(tmp)  # tmp: (128, 20, 20)
        # 5. PAD Mask(only for Multi-Head Attention Layer)
        if sub_sequence_mask == None: # for Multi-Head Layer
            tmp_pad_mask = padding_mask.unsqueeze(2).repeat(1,1,l) # tmp_pad_mask: (128, 20, 20)
            tmp = tmp * tmp_pad_mask # row-wise padding
            tmp = tmp * tmp_pad_mask.transpose(1,2) # column-wise padding
        # 6. Matmul with Value
        out = torch.bmm(tmp, V)  # out: (128, 20, 64)

        return out

class Decoder(nn.Module):
    def __init__(self, dim_model, d_k, d_v, n_head, dim_hidden, d_prob, n_dec_layer):
        super().__init__()
        self.dim_model = dim_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.dim_hidden = dim_hidden
        self.d_prob = d_prob
        self.n_dec_layer = n_dec_layer
        self.dropout_layer = nn.Dropout(p=d_prob)

        self.MaskedMultiHeadLayer = MaskedMultiHead(dim_model, dim_hidden, n_head, d_k, d_v)
        self.MultiHeadAttentionLayer = MultiHeadAttention(dim_model, dim_hidden, n_head, d_k, d_v)
        self.FeedForwardLayer = FeedForward(dim_model, dim_hidden)
        self.layernorm_layer = nn.LayerNorm(dim_model)

    def forward(self, enc_out, dec_inp, padding_mask):
        # enc_out: (128, 20, 512), dec_inp: (128, 21, 512)
        dec_max_len = dec_inp.shape[1]
        out = dec_inp[:,:enc_out.shape[1],:] # 맨 마지막 빼버리기

        # Make sub_sequence_mask for advance
#         sub_seq_mask = torch.ones(dec_max_len, dec_max_len)

#         a = torch.arange(dec_max_len)+1
#         a = a.unsqueeze(0)
#         a = a.repeat(dec_max_len, 1)

#         b = torch.arange(dec_max_len)+1
#         b = b.unsqueeze(1)
#         b = b.repeat(1, dec_max_len)

#         c = (a <= b)

#         sub_seq_mask = sub_seq_mask * c
        sub_seq_mask = masked_attn_mask(dec_max_len)

        for idx in range(self.n_dec_layer):
            inp = out

            # 1. Masked Multi-Head Attention Layer
            out = self.MaskedMultiHeadLayer(out, padding_mask, sub_seq_mask)
            # Dropout
            out = self.dropout_layer(out)

            # Add
            out = inp + out
            # Norm
            out = self.layernorm_layer(out)

            inp = out
            # 2. Multi-Head Attention Layer
            out = self.MultiHeadAttentionLayer(enc_out, padding_mask, dec_inp=inp)
            # Dropout
            out = self.dropout_layer(out)

            # Add
            out = inp + out  # (2560, 512)
            # Norm
            out = self.layernorm_layer(out)

            inp = out
            # 3. Feed Forward Layer
            out = self.FeedForwardLayer(out)
            # Dropout
            out = self.dropout_layer(out)

            # Add
            out = inp + out
            # Norm
            out = self.layernorm_layer(out)

        return out

class MaskedMultiHead(nn.Module):
    def __init__(self, dim_model, dim_hidden, n_head, d_k, d_v):
        super().__init__()
        self.dim_model = dim_model
        self.dim_hidden = dim_hidden
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_o = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_head * d_k, dim_model)))  # w_o: (512, 512)

    def forward(self, src, padding_mask, sub_seq_mask):
        # padding_mask: (128, 21)
        out_list = []

        for idx in range(self.n_head):
            attention_layer = Attention(self.dim_model, self.dim_hidden, self.d_k, self.d_v)
            out = attention_layer(src, padding_mask, sub_sequence_mask=sub_seq_mask) # out: (128, 21, 64)
            out_list.append(out)

        out_cat = torch.cat(out_list, dim=2)  # out_cat: (2560, 512)
        res = torch.matmul(out_cat, self.w_o)  # res: (2560, 512)

        return res
