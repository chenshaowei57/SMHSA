# -*- coding: utf-8 -*-
# follow NCRF++, a Neural Sequence Labeling Toolkit.

import torch
import torch.nn as nn
import numpy as np
from charbilstm import CharBiLSTM
from charcnn import CharCNN

class WordEmbedding(nn.Module):
    def __init__(self, args, model_params):
        super(WordEmbedding, self).__init__()
        self.gpu = args.ifgpu
        self.use_char = args.useChar
        self.char_hidden_dim = args.char_hidden_dim
        self.char_embedding_dim = args.char_embedding_dim
        self.embedding_dim = model_params.embedding_dim
        self.drop = nn.Dropout(args.dropout)
    #char Embedding
        if self.use_char:
            if args.charExtractor == "CNN":
                self.char_feature = CharCNN(model_params.char_alphabet.size(), model_params.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, args.dropout, self.gpu)
            elif args.charExtractor == "LSTM":
                self.char_feature = CharBiLSTM(model_params.char_alphabet.size(), model_params.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, args.dropout, self.gpu)
            else:
                print("Error char feature selection, please check parameter data.char_feature_extractor (CNN/LSTM).")
                exit(0)

    #word Embedding
        self.word_embedding = nn.Embedding(model_params.word_alphabet.size(), self.embedding_dim)
        if model_params.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(model_params.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(model_params.word_alphabet.size(), self.embedding_dim)))





    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def forward(self, word_inputs, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs = self.word_embedding(word_inputs)
        word_list = [word_embs]

        if self.use_char:
            ## calculate char lstm last hidden
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            ## concat word and char together
            word_list.append(char_features)

        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent
