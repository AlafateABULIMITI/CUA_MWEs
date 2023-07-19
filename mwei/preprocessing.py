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
        self.ignores = []
        self.sentences = self.get_sentences()

    @property
    def max_length(self):
        max_ = [len(sent) for sent in self.sentences]
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
        sentences = []
        for idx, item in enumerate(self.tsv_):
            sent = Sentence(idx, item)
            if (
                self.length_threshold is not " "
                and len(sent) <= self.length_threshold
                or self.length_threshold is " "
            ):
                sentences.append(sent)
            else:
                self.ignores.append(idx)
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
        return [
            [token["FORM"] for token in self.item.words if "-" not in token["ID"]]
        ]

    @property
    def vocabs(self):
        return set(self.tokens[0])

    @property
    def _pos_s(self):
        return [token["UPOS"] for token in self.item.words]

    @property
    def _mwe_labels(self):
        return [token["PARSEME:MWE"] for token in self.item.words]

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
        for pos in self._pos_s:
            vec = np.zeros((1, 18), dtype=np.int)
            index = POS_DIC.get(pos)
            vec[0][index] = 1
            pos_v = np.append(pos_v, vec, axis=0)
        return pos_v

    def _bert_vec(self, client, bert):
        return bert.vectorize(client, self.tokens)

    def get_vector(self, client, bert):
        pos_v = self._pos_vec
        bert_v = self._bert_vec(client, bert)
        return np.hstack((bert_v, pos_v))


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
        train = []
        counter = 0
        with BertServer(bert.start_args):
            with BertClient() as client:
                if corpus_type == "test":
                    test = []
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
                elif corpus_type == "train":
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
                            save_to = f"{save_to}_{counter}"
                            np.save(save_to, np.asarray(train))
                            train = []
        logging.info("training set is done ===========")

    def save_vocabs(self, save_to):
        vocabs = self.corpus.vocabs
        with open(save_to, "w+") as wf:
            for vocab in vocabs:
                wf.write(vocab + "\n")

    def get_vocabs(self, to_save):
        vocabs_dic = {}
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
        index_list = []
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

