# -*- coding: utf-8 -*-
# @Author: Shaowei Chen,   Contact: chenshaowei0507@163.com
# @Date:   2020-4-27

import sys
import argparse
import torch
from alphabet import Alphabet
sys.path.append("../")

word_alphabet = Alphabet('word', True)
label_alphabet = Alphabet('label', True)
label_alphabet.add("O")
label_alphabet.add("B")
label_alphabet.add("I")
relation_alphabet = Alphabet('relation', True)
char_alphabet = Alphabet('char', True)


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


def readDataFromFile(path):
    f = open(path, "r", encoding="utf-8")
    lines = f.readlines()
    f.close()

    seq = 0
    char_seq = 0
    datasets = []
    words = []
    labels = []
    relations = []

    for l in lines:
        if l.strip() == "#Relations":
            continue
        elif l.strip() == "" and len(words) > 0:
            datasets.append({"words": words, "labels": labels, "relations": relations})
            if len(words) > seq:
                seq = len(words)
            words = []
            labels = []
            relations = []
        elif len(l.strip().split("\t")) == 2:
            tempLine = l.strip().split("\t")
            if len(tempLine[0]) > char_seq:
                char_seq = len(tempLine[0])
            words.append(tempLine[0].lower())
            labels.append(tempLine[1])
        elif len(l.strip().split("\t")) == 5:
            rel = l.strip().split("\t")
            rel[:4] = list(map(int, rel[0: 4]))
            relations.append(rel)
    return datasets, seq, char_seq


def convert_examples(examples, max_seq_length=10, max_char_length=10):
    features = []
    for (example_index, example) in enumerate(examples):
        tokens = []
        token_ids = []
        chars = []
        char_ids = []
        char_mask = []
        charLength = []
        tokenLength = []
        labels = []
        label_ids = []
        relations = []
        #### split words and labels ####
        for (i, token) in enumerate(example["words"]):
            char = []
            charId = []
            # token
            tokens.append(token)
            word_alphabet.add(token)
            token_ids.append(word_alphabet.get_index(token))
            # char
            for w in token:
                char.append(w)
                char_alphabet.add(w)
                charId.append(char_alphabet.get_index(w))
            assert len(charId) <= max_char_length
            charLength.append(len(charId))
            # mask char
            char_m = [1] * len(charId)
            while len(charId) < max_char_length:
                charId.append(0)
                char_m.append(0)
            char_mask.append(char_m)
            char_ids.append(charId)
            chars.append(char)
            # label
            label = example["labels"][i]
            labels.append(label)
            label_ids.append(label_alphabet.get_index(label))
        # relation
        gold_relations = example["relations"]
        for gr in gold_relations:
            relation_alphabet.add(gr[4])
            relations.append([gr[0], gr[1], gr[2], gr[3], relation_alphabet.get_index(gr[4])])
        assert len(tokens) <= max_seq_length


        # mask token
        token_mask = [1] * len(tokens)
        tokenLength.append(len(tokens))
        while len(token_ids) < max_seq_length:
            token_ids.append(0)
            label_ids.append(0)
            token_mask.append(0)
            char_mask.append([0] * max_char_length)
            char_ids.append([0] * max_char_length)
            charLength.append(0)

        features.append(
            InputFeatures(
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
                gold_relations))
    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="./data/NYT/train.txt")
    parser.add_argument("--dev_file", type=str, default="./data/NYT/dev.txt")
    parser.add_argument("--test_file", type=str, default="./data/NYT/test.txt")
    parser.add_argument("--output_file", type=str, default="./data/NYT/NYT.pt")
    args = parser.parse_args()

    train_set, train_max_length, train_max_char_length = readDataFromFile(args.train_file)
    dev_set, dev_max_length, dev_max_char_length = readDataFromFile(args.dev_file)
    test_set, test_max_length, test_max_char_length = readDataFromFile(args.test_file)

    # the max length should be modified according to the dataset
    train_features = convert_examples(train_set, max_seq_length=train_max_length, max_char_length=train_max_char_length)
    dev_features = convert_examples(dev_set, max_seq_length=dev_max_length, max_char_length=dev_max_char_length)
    test_features = convert_examples(test_set, max_seq_length=test_max_length, max_char_length=test_max_char_length)

    torch.save({"train": train_features, "test": test_features, "dev": dev_features, "word_alpha": word_alphabet,
                "label_alpha": label_alphabet,
                "relation_alpha": relation_alphabet, "char_alpha": char_alphabet}, args.output_file)
