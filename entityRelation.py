# -*- coding: utf-8 -*-
# @Author: Shaowei Chen,   Contact: chenshaowei0507@163.com
# @Date:   2020-4-27

import torch
import torch.nn as nn
from relationAttention import RelationAttention
from crf import CRF
from wordEmbedding import WordEmbedding
from wordHiddenRep import WordHiddenRep


class entityRelation(nn.Module):
    def __init__(self, args, model_params):
        super(entityRelation, self).__init__()
        print("build network...")
        print("bbb")
        self.gpu = args.ifgpu
        self.label_size = model_params.label_alphabet.size()
        self.bert_encoder_dim = args.encoder_dim
        self.targetHiddenDim = args.targetHiddenDim
        self.relationHiddenDim = args.relationHiddenDim
        self.relation_num = args.relationNum
        self.drop = args.dropout
        # buliding model
        # encoding layer
        self.Embedding = WordEmbedding(args, model_params)
        self.encoder = WordHiddenRep(args, model_params)
        # module linear
        self.u_input_Linear = nn.Linear(self.bert_encoder_dim, self.targetHiddenDim)
        self.r_input_Linear = nn.Linear(self.bert_encoder_dim, self.relationHiddenDim)
        # Tag Linear
        self.targetHidden2Tag = nn.Linear(self.targetHiddenDim, self.label_size + 2)
        # CRF
        self.crf = CRF(self.label_size, self.gpu)
        # Relation
        self.relationAttention = RelationAttention(args)
        # Dropout
        self.dropout = nn.Dropout(self.drop)

        if self.gpu:
            self.Embedding = self.Embedding.cuda()
            self.encoder = self.encoder.cuda()
            self.u_input_Linear = self.u_input_Linear.cuda()
            self.r_input_Linear = self.r_input_Linear.cuda()
            self.targetHidden2Tag = self.targetHidden2Tag.cuda()
            self.crf = self.crf.cuda()
            self.relationAttention = self.relationAttention.cuda()
            self.dropout = self.dropout.cuda()

    def neg_log_likelihood_loss(self, all_input_ids, input_length, all_input_mask, all_char_ids,
                                char_length, char_recover, all_relations, all_labels):

        batch_size = all_input_ids.size(0)
        seq_len = all_input_ids.size(1)

        targetPredictScore, R_tensor = self.mainStructure(all_input_ids, input_length, all_input_mask, all_char_ids,
                                                          char_length, char_recover)

        target_loss = self.crf.neg_log_likelihood_loss(targetPredictScore, all_input_mask.byte(), all_labels) / (batch_size)
        scores, tag_seq = self.crf._viterbi_decode(targetPredictScore, all_input_mask.byte())

        relationScale = all_relations.transpose(1, 3).contiguous().view(-1, self.relation_num)
        relation_loss_function = nn.BCELoss(size_average=False)
        relationScoreLoss = R_tensor.transpose(1, 3).contiguous().view(-1, self.relation_num)
        relation_loss = relation_loss_function(relationScoreLoss, relationScale.float()) / (batch_size * seq_len)

        return target_loss, relation_loss, tag_seq, R_tensor

    def forward(self, all_input_ids, input_length, all_input_mask, all_char_ids, char_length, char_recover):

        targetPredictScore, R_tensor = self.mainStructure(all_input_ids, input_length, all_input_mask, all_char_ids,
                                                          char_length, char_recover)
        scores, tag_seq = self.crf._viterbi_decode(targetPredictScore, all_input_mask.byte())

        return tag_seq, R_tensor

    def mainStructure(self, all_input_ids, input_length, all_input_mask, all_char_ids, char_length, char_recover):
        batch_size = all_input_ids.size(0)
        seq_len = all_input_ids.size(1)

        # encoding layer
        wordEmbedding = self.Embedding(all_input_ids, all_char_ids, char_length, char_recover)
        maskEmb = all_input_mask.view(batch_size, seq_len, 1).repeat(1, 1, wordEmbedding.size(2))
        wordEmbedding = wordEmbedding * (maskEmb.float())
        sequence_output = self.encoder(wordEmbedding, input_length)

        # module linear
        h_t = self.u_input_Linear(sequence_output)
        h_r = self.r_input_Linear(sequence_output)

        # entity extraction module
        targetPredictInput = self.targetHidden2Tag(h_t)

        # relation detection module
        relationScore = self.relationAttention(h_r)

        return targetPredictInput, relationScore
