# -*- coding: utf-8 -*-
# @Author: Shaowei Chen,   Contact: chenshaowei0507@163.com
# @Date:   2020-4-27

import numpy as np
import sys
import torch
import argparse


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 tokens,
                 token_ids,
                 token_mask,
                 chars,
                 char_ids,
                 char_mask,
                 charLength,
                 tokenLength,
                 labels,
                 label_ids,
                 relations,
                 gold_relations):
        self.tokens = tokens
        self.token_ids = token_ids
        self.token_mask = token_mask
        self.tokenLength = tokenLength
        self.labels = labels
        self.label_ids = label_ids
        self.relations = relations
        self.gold_relations = gold_relations
        self.chars = chars
        self.char_ids = char_ids
        self.char_mask = char_mask
        self.charLength = charLength


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if embedd_dim + 1 != len(tokens):
                    continue
                else:
                    assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
    return pretrain_emb, embedd_dim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_emb_dir", type=str, default="./data/glove.840B.300d.txt")
    parser.add_argument("--data_path", type=str, default="./data/NYT/NYT.pt")
    parser.add_argument("--output_path", type=str, default="./data/NYT/NYT_embedding.pt")
    args = parser.parse_args()

    data = torch.load(args.data_path)
    word_alphabet = data["word_alpha"]
    word_embedding, word_emb_dim = build_pretrain_embedding(args.word_emb_dir, word_alphabet, 300, False)
    torch.save({"preTrainEmbedding": word_embedding, "emb_dim": word_emb_dim}, args.output_path)
