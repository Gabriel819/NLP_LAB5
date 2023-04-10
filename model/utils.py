import torch.nn as nn
import torch

def pad_mask(src):
    return src != 2
    # As 'PAD' token's index is 2, get tensor of what is not padding. It will have a form of tensor([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
    # True,  True,  True,  True, False, False, False, False, False, False]), which means True: not pad, False: pad token.

def masked_attn_mask(dec_max_len):
    sub_seq_mask = torch.ones(dec_max_len, dec_max_len)

    a = torch.arange(dec_max_len) + 1
    a = a.unsqueeze(0)
    a = a.repeat(dec_max_len, 1)

    b = torch.arange(dec_max_len) + 1
    b = b.unsqueeze(1)
    b = b.repeat(1, dec_max_len)

    c = (a <= b)

    sub_seq_mask = sub_seq_mask * c

    return sub_seq_mask