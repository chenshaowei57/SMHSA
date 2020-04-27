# -*- coding: utf-8 -*-
# @Author: Shaowei Chen,   Contact: chenshaowei0507@163.com
# @Date:   2020-4-27

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class RelationAttention(nn.Module):
    def __init__(self, args):
        super(RelationAttention, self).__init__()
        self.gpu = args.ifgpu
        self.hd = args.relationHiddenDim
        self.head_dim = args.relationHeadDim
        self.num_heads = args.relationNum
        self.scaling1 = (self.head_dim) ** (-0.5)

        self.q_linear = nn.Linear(self.hd, self.num_heads * self.head_dim, bias=True)
        self.k_linear = nn.Linear(self.hd, self.num_heads * self.head_dim, bias=True)

        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0)
        nn.init.constant_(self.k_linear.bias, 0)


    def forward(self, relationDecoderResult):

        batchSize = relationDecoderResult.size(0)
        seqLen = relationDecoderResult.size(1)

        query = relationDecoderResult.transpose(0,1)
        key = relationDecoderResult.transpose(0,1)

        q = self.q_linear(query)
        k = self.k_linear(key)

        q = q.contiguous().view(seqLen, batchSize * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(seqLen, batchSize * self.num_heads, self.head_dim).transpose(0, 1)
        q *= self.scaling1

        attn_weights = torch.bmm(q, k.transpose(1, 2)).view(batchSize, self.num_heads, seqLen, seqLen)
        attn_weights = F.softmax(attn_weights, dim=3)
        return attn_weights