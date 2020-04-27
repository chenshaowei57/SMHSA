# -*- coding: utf-8 -*-
# follow NCRF++, a Neural Sequence Labeling Toolkit.

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class WordHiddenRep(nn.Module):
    def __init__(self, args, model_params):
        super(WordHiddenRep, self).__init__()
        print("build word embedding...")
        self.use_char = args.useChar
        self.input_size = model_params.embedding_dim# + data.char_hidden_dim
        if self.use_char:
            self.input_size += args.char_hidden_dim
        self.drop = nn.Dropout(args.dropout)

        if args.encoder_Bidirectional:
            self.hidden_dim = args.encoder_dim // 2
        else:
            self.hidden_dim = args.encoder_dim

    # word hidden rep
        if args.encoderExtractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=args.encoder_Bidirectional)
        else:
            print("Error char feature selection, please check parameter data.char_feature_extractor (LSTM).")
            exit(0)



    def forward(self, embedding_represent, word_seq_lengths):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        packed_words = pack_padded_sequence(embedding_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        feature_out = self.drop(lstm_out)
        return feature_out
