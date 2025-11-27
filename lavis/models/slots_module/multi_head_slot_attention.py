from __future__ import annotations

import torch
from torch import einsum, nn
from torch.nn import init
from torch.nn import Module
import torch.nn.functional as F

from einops import pack, unpack
from einops.layers.torch import Rearrange
from lavis.models.slots_module.adaptive_slot_wrapper import AdaptiveSlotWrapper
from einops import rearrange

class MultiHeadSlotAttention(Module):
    # B,L,D->B,S,D
    def __init__(self, num_slots, input_dim, emb_dim, heads = 12, 
                 iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = emb_dim ** -0.5

        self.slots = nn.Parameter(torch.zeros(1, num_slots, emb_dim))
        init.xavier_uniform_(self.slots)

        self.norm_input  = nn.LayerNorm(input_dim)
        self.norm_slots  = nn.LayerNorm(emb_dim)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.to_q = nn.Linear(emb_dim, emb_dim)
        self.to_k = nn.Linear(input_dim, emb_dim)
        self.to_v = nn.Linear(input_dim, emb_dim)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = nn.Linear(emb_dim, emb_dim)

        self.gru = nn.GRUCell(emb_dim, emb_dim)

        hidden_dim = max(emb_dim, hidden_dim)

        self.norm_pre_ff = nn.LayerNorm(emb_dim)

        # coupling function
        self.cf = CouplingFunction(num_slots=num_slots,
                            input_dim=emb_dim,
                            output_dim=emb_dim)

    def forward(self, inputs):
        b = inputs.size(0)

        slots = self.slots.expand(b, -1, -1)
        inputs = self.norm_input(inputs)        

        k, v = self.to_k(inputs), self.to_v(inputs)
        k, v = map(self.split_heads, (k, v))
        
        for _ in range(self.iters):
            slots_prev = slots.clone()
            slots = self.norm_slots(slots)

            q = self.to_q(slots)
            q = self.split_heads(q)

            dots = einsum('... i d, ... j d -> ... i j', q, k) * self.scale

            attn = dots.softmax(dim = -2)
            attn = F.normalize(attn + self.eps, p = 1, dim = -1)

            updates = einsum('... j d, ... i j -> ... i d', v, attn)
            updates = self.merge_heads(updates)
            updates = self.combine_heads(updates)

            updates, packed_shape = pack([updates], '* d')
            slots_prev, _ = pack([slots_prev], '* d')

            slots = self.gru(updates, slots_prev)

            slots, = unpack(slots, packed_shape, '* d')
            slots = slots + self.cf(self.norm_pre_ff(slots))

        return slots


class CouplingFunction(Module):
    def __init__(self, num_slots: int, input_dim: int, output_dim: int):
        super().__init__()
        self.trans = nn.Linear(input_dim, output_dim)
        self.sig = nn.Sequential(nn.Linear(input_dim, num_slots),
                                 nn.Sigmoid())
    
    def forward(self, slots):
        fusion_weights = self.sig(slots) # 1,L,L
        slots = torch.matmul(fusion_weights, slots)
        slots = self.trans(slots)
        return slots


class SlotFusion(Module):
    def __init__(self, num_slots, input_dim, emb_dim, heads = 12, iters = 3, eps = 1e-8, 
                 hidden_dim = 128, use_adapt = False, tau = 1.):
        super().__init__()
        self.slot_module = MultiHeadSlotAttention(num_slots, input_dim, emb_dim, heads, iters, eps, hidden_dim)
        self.use_adapt = use_adapt
        if use_adapt:
            self.slot_module = AdaptiveSlotWrapper(self.slot_module, tau)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        if self.use_adapt:
            slot_feats, slot_attn = self.slot_module(inputs)
            return slot_feats, slot_attn
        else:
            slot_feats = self.slot_module(inputs)
            return slot_feats
