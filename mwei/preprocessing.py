import os
import sys

sys.path.append("./utils")
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
import tsvlib
from pprint import pprint
from bert_serving.server import BertServer
from bert_serving.client import BertClient
from bert import BertWordEmbedding

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
    def __init__(self, corpus):
        self.corpus = corpus

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

    @property
    def sentences(self):
        sentences = list()
        for idx, item in enumerate(self.tsv_):
            sentences.append(Sentence(self.corpus, idx, item))
        return sentences

    @property
    def tsv_(self):
        with open(self.corpus, "r+") as f:
            tsv_ = list(tsvlib.iter_tsv_sentences(f))
        return tsv_

    @property
    def max_length_sent(self):
        lengths = [sent.length for sent in self.sentences]
        return max(lengths)

    @property
    def vocabs(self):
        vocabs = set()
        for sent in self.sentences:
            vocabs = vocabs.union(sent.vocabs)
        return vocabs


class Sentence(Corpus):
    def __init__(self, corpus, sent_id, item):
        super().__init__(corpus)
        self.sent_id = sent_id
        self.item = item
        self.length = len(self.tokens[0])

    @property
    def tokens(self):
        tokens = [[token["FORM"] for token in self.item.words]]
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

    def save_train(self, save_to):
        max_length = self.corpus.max_length_sent
        sentences = self.corpus.sentences
        bert = BertWordEmbedding()
        # train = np.empty((0, 53, 786))
        train = list()
        # print(embedding.shape)
        with BertServer(bert.start_args):
            with BertClient() as client:
                for sent in sentences:
                    vector = sent.get_vector(client, bert)
                    labels = np.pad(
                        sent.labels,
                        (0, max_length - vector.shape[0]),
                        mode="constant",
                        constant_values=0,
                    )  # padding  0.
                    # for solving the broadcast problem
                    labels = np.expand_dims(labels, axis=0)
                    vector = np.pad(
                        vector,
                        [(0, max_length - vector.shape[0]), (0, 0)],
                        mode="constant",
                        constant_values=0,
                    )  # padding  0.
                    train.append((vector, labels))
                    # vector = np.expand_dims(vector, axis=0)
                    # print(vector.shape)
                    # train = np.concatenate((train, vector), axis=0)
        # print(embedding.shape)
        # TODO show to the prof.
        # with open("../tmp/log_embedding.txt", "w+") as f:
        #     print(train, file=f)

        np.save(save_to, train)  # np.load(outfile, allow_pickle=True)
        return train

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
    corpus = "../data/FR/train_test.cupt"
    cps = Corpus(corpus)
    print(len(cps.vocabs))
    # print(cps.max_length_sent)
    # sentences = cps.sentences
    # bert = BertWordEmbedding()
    # log = open("log.txt", "a+")
    # with BertServer(bert.start_args):
    #     with BertClient() as client:
    #         for idx, item in enumerate(cps.tsv_):
    #             sent = Sentence(corpus, idx, item)
    #             vector = sent.get_vector(client, bert)
    #             print("Tokens:\n", sent.tokens, file=log)
    #             print("Len of Tokens:\n", len(sent.tokens[0]), file=log)
    #             print("MWE labels:\n", sent.labels, file=log)
    #             print("label length:\n", len(sent.labels), file=log)
    #             print("Vector shape:\n", vector.shape, file=log)
    pre = Preprocessing(cps)
    save_to = "../tmp/train_seq2seq/train"
    pre.save_train(save_to)
    c = np.load("../tmp/train_seq2seq/train.npy", allow_pickle=True)
    print(isinstance(c, list))
    print(c[0][1])
    # pre.save_vocabs(save_to)
    # dic = pre.get_vocabs(save_to)
    # print(type(dic))
    # print(dic)

