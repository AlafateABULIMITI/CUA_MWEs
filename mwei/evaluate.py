import sys
from collections import Counter
from shutil import copyfile

import keras.backend as K
import numpy as np
from bert_serving.client import BertClient
from bert_serving.server import BertServer
from conllu import parse_incr
from keras.layers import Activation, Dense, TimeDistributed
from keras.models import Sequential
from seq2seq import AttentionSeq2Seq

sys.path.append("./utils")
import tsvlib
from bert import BertWordEmbedding
from preprocessing import Corpus


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


class Evaluation:
    def __init__(
        self, eval_corpus, eval_corpus_vectors_path, result, model_path, model_config
    ):
        super().__init__()
        self.eval_corpus = eval_corpus
        self.eval_corpus_vectors_path = eval_corpus_vectors_path
        self.result = result
        self.model_path = model_path
        self.model_config = model_config

        self.model = Sequential()
        self.model.add(
            AttentionSeq2Seq(
                output_dim=model_config.get("output_dim"),
                hidden_dim=model_config.get("hidden_dim"),
                output_length=model_config.get("output_length"),
                input_shape=(
                    model_config.get("input_length"),
                    model_config.get("input_dim"),
                ),
            )
        )
        self.model.add(TimeDistributed(Dense(model_config.get("output_dim"))))
        self.model.add(Activation("softmax"))
        self.model.load_weights(model_path)

    @property
    def eval_corpus_vectors(self):
        vecs = np.load(self.eval_corpus_vectors_path, allow_pickle=True)
        print(vecs.shape)
        return vecs

    def _predict(self, vectors):
        mwei_preds_list = list()
        preds = self.model.predict(vectors)
        predicts = K.get_value(K.argmax(preds, axis=2))
        for pred in predicts:
            mwei_preds_list.append(self._transform_preds(pred))
        return mwei_preds_list

    def _transform_preds(self, preds):
        counter = 0
        mweis_preds = ["*"] * len(preds)
        for idx, val in enumerate(preds):
            if val == 1:
                mweis_preds[idx] = str(counter + 1)
                counter = counter + 1
            elif val == 2:
                mweis_preds[idx] = str(-1)

        for idx, val in enumerate(mweis_preds):
            if val == str(-1):
                for i in range(idx, -1, -1):
                    if mweis_preds[i].isnumeric():
                        mweis_preds[idx] = mweis_preds[i]
                        break
        return mweis_preds

    # TODO: the problem 15-17
    # ? verification
    def write(self, eval_corpus_path):
        mwei_preds_list = self._predict(self.eval_corpus_vectors)
        print(mwei_preds_list)
        with open(eval_corpus_path, "r", encoding="utf-8") as data_file:
            with open(self.result, "w+", encoding="utf-8") as save_to:
                for sent_idx, tokenlist in enumerate(parse_incr(data_file)):
                    for preds in mwei_preds_list:
                        if sent_idx not in self.eval_corpus.ignores:
                            for idx, token in enumerate(tokenlist):
                                # print(token)
                                if isinstance(token["id"], tuple):
                                    token["parseme:mwe"] = "*"
                                else:
                                    token["parseme:mwe"] = preds[token["id"] - 1]
                        else:
                            for idx, token in enumerate(tokenlist):
                                token["parseme:mwe"] = "*"
                    a = tokenlist.serialize()
                    save_to.write(a)
        return a


if __name__ == "__main__":
    model_config = {
        "input_length": 100,
        "input_dim": 786,
        "output_length": 100,
        "output_dim": 3,
        "hidden_dim": 512,
    }
    eval_corpus_path = "../data/FR/test.blind.cupt"
    # eval_corpus_path = "../tmp/train_test.cupt"
    result_path = "../data/test/test.blind_result.cupt"
    cps = Corpus(eval_corpus_path, "test", 100)
    model_path = "../models/test/model_3000.h5"
    # evaluation
    # vector_path = "../tmp/test_train_50.npy"
    test_vector_path = "../data/train_seq2seq/test.blind.npy"
    a = Evaluation(cps, test_vector_path, result_path, model_path, model_config)
    print(a.write(eval_corpus_path))
    # print(a.write())
