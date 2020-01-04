import os
import sys
import logging

sys.path.append("./utils")

import numpy as np
from bert_serving.client import BertClient
from bert_serving.server import BertServer

import tsvlib
from bert import BertWordEmbedding

np.set_printoptions(threshold=sys.maxsize)

POS_DIC = {
    "ADJ": 0,
    "ADV": 1,
    "INTJ": 2,
    "NOUN": 3,
    "PROPN": 4,
    "VERB": 5,
    "ADP": 6,
    "AUX": 7,
    "CCONJ": 8,
    "DET": 9,
    "NUM": 10,
    "PART": 11,
    "PRON": 12,
    "SCONJ": 13,
    "PUNCT": 14,
    "SYM": 15,
    "X": 16,
    "UNK": 17,
}


class Corpus:
    def __init__(self, corpus, corpus_type, length_threshold=" "):
        self.corpus = corpus
        self.corpus_type = corpus_type
        self.length_threshold = length_threshold
        self.ignores = list()
        self.sentences = self.get_sentences()

    @property
    def max_length(self):
        max_ = list()
        for sent in self.sentences:
            max_.append(len(sent))
        return max(max_)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, key):
        if 0 <= key <= len(self.corpus):
            return self.sentences[key]

    def __setitem__(self, key, sentence):
        if 0 <= key <= len(self.corpus):
            self.sentences[key] = sentence

    def __delitem__(self, key):
        if 0 <= key <= len(self.corpus):
            del self.sentences[key]


    def get_sentences(self):
        sentences = list()
        for idx, item in enumerate(self.tsv_):
            sent = Sentence(idx, item)
            if self.length_threshold is not " ":
                if len(sent) <= self.length_threshold:
                    sentences.append(sent)
                else:
                    self.ignores.append(idx)
            else:
                sentences.append(sent)
        return sentences

    @property
    def tsv_(self):
        with open(self.corpus, "r+") as f:
            tsv_ = list(tsvlib.iter_tsv_sentences(f))
        return tsv_

    @property
    def vocabs(self):
        vocabs = set()
        for sent in self.sentences:
            vocabs = vocabs.union(sent.vocabs)
        return vocabs


class Sentence:
    def __init__(self, sent_id, item):
        self.sent_id = sent_id
        self.item = item

    def __len__(self):
        return len(self.tokens[0])

    @property
    def tokens(self):
        tokens = [
            [token["FORM"] for token in self.item.words if "-" not in token["ID"]]
        ]
        return tokens

    @property
    def vocabs(self):
        vocabs = set(self.tokens[0])
        return vocabs

    @property
    def _pos_s(self):
        pos_s = [token["UPOS"] for token in self.item.words]
        return pos_s

    @property
    def _mwe_labels(self):
        labels = [token["PARSEME:MWE"] for token in self.item.words]
        return labels

    @property
    def labels(self):
        mwe_labels = self._mwe_labels

        def pattern(s):
            if s == "*":
                return 0
            elif s.endswith(":VID"):
                return 1
            else:
                return 2

        mwe_labels = np.asarray(list(map(pattern, mwe_labels)))
        return mwe_labels

    @property
    def _pos_vec(self):
        pos_v = np.empty((0, 18), dtype=np.int)
        for idx, pos in enumerate(self._pos_s):
            vec = np.zeros((1, 18), dtype=np.int)
            index = POS_DIC.get(pos)
            vec[0][index] = 1
            pos_v = np.append(pos_v, vec, axis=0)
        return pos_v

    def _bert_vec(self, client, bert):
        bert_v = bert.vectorize(client, self.tokens)
        return bert_v

    def get_vector(self, client, bert):
        pos_v = self._pos_vec
        bert_v = self._bert_vec(client, bert)
        vector = np.hstack((bert_v, pos_v))
        return vector


class Preprocessing:
    def __init__(self, corpus):
        self.corpus = corpus

    def save_vectors(self, save_to):
        max_length = self.corpus.max_length
        corpus_type = self.corpus.corpus_type
        length_threshold = self.corpus.length_threshold
        length = len(self.corpus)
        sentences = self.corpus.sentences
        bert = BertWordEmbedding()
        train = list()
        counter = 0
        with BertServer(bert.start_args):
            with BertClient() as client:
                if corpus_type == "train":
                    for sent in sentences:
                        vector = sent.get_vector(client, bert)
                        labels = np.pad(
                            sent.labels,
                            (0, length_threshold - vector.shape[0]),
                            mode="constant",
                            constant_values=0,
                        )  # padding  0.
                        # for solving the broadcast problem
                        labels = np.expand_dims(labels, axis=0)
                        vector = np.pad(
                            vector,
                            [(0, length_threshold - vector.shape[0]), (0, 0)],
                            mode="constant",
                            constant_values=0,
                        )  # padding  0.
                        train.append((vector, labels))
                        counter += 1
                        logging.info(f"sentence id:=================={counter}")
                        if counter % 2000 == 0 or counter == length:
                            logging.info(f"training set{counter} is done ===========")
                            save_to = save_to + "_" + str(counter)
                            np.save(save_to, np.asarray(train))
                            train = list()
                elif corpus_type == "test":
                    test = list()
                    for sent in sentences:
                        vector = sent.get_vector(client, bert)
                        vector = np.pad(
                            vector,
                            [(0, length_threshold - vector.shape[0]), (0, 0)],
                            mode="constant",
                            constant_values=0,
                        )
                        test.append(vector)
                    test = np.asarray(test)
                    np.save(save_to, test)
                    # TODO: break for just 3000

        logging.info("training set is done ===========")

    def save_vocabs(self, save_to):
        vocabs = self.corpus.vocabs
        with open(save_to, "w+") as wf:
            for vocab in vocabs:
                wf.write(vocab + "\n")

    def get_vocabs(self, to_save):
        vocabs_dic = dict()
        with open(to_save, "r") as rf:
            lines = rf.readlines()
            for idx, line in enumerate(lines):
                vocabs_dic[line.replace("\n", "")] = idx + 1
        return vocabs_dic

    def save_train_with_index(self, dic, save_to):
        with open(save_to, "w+") as wf:
            sentences = self.corpus.sentences
            max_length = self.corpus.max_length_sent
            for sentence in sentences:
                tokens = sentence.tokens[0]
                idx_tokens = self.tokens_2_index(dic, tokens)
                idx_tokens = self.pad(idx_tokens, max_length)
                # print(len(idx_tokens))
                idx_tokens = list(map(lambda x: str(x), idx_tokens))
                labels = sentence.labels
                labels = self.pad(labels, max_length)
                labels = list(map(lambda x: str(x), labels))
                print(len(labels))
                idx_tokens_str = " ".join(idx_tokens)
                # print(idx_tokens_str)
                labels_str = " ".join(labels)
                wf.write(idx_tokens_str + "\t" + labels_str + "\n")

    def tokens_2_index(self, dic, tokens):
        index_list = list()
        for token in tokens:
            idx = dic.get(token)
            index_list.append(idx)
        return index_list

    def pad(self, l, max_length):
        zeros = [0] * (max_length - len(l))
        l.extend(zeros)
        return l


if __name__ == "__main__":
    MAX_LENGTH = 100
    corpus = "../data/FR/test.blind.cupt"
    # corpus = "../tmp/train_test.cupt"

    corpus_type = "test"
    cps = Corpus(corpus, corpus_type, MAX_LENGTH)
    print(cps.ignores)
    print(len(cps))
    pre = Preprocessing(cps)
    # save_to = "../data/train_seq2seq/train"
    save_to = "../data/train_seq2seq/test.blind"
    pre.save_vectors(save_to)

