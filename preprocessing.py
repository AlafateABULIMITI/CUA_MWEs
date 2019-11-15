import os
import sys

sys.path.append("./utils")
import tsvlib


class Preprocessing:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.sentences = self._give_sentences()

    def _give_sentences(self):
        sentences = list()
        corpus_file = open(self.corpus_path)
        tsv_ = list(tsvlib.iter_tsv_sentences(corpus_file))
        for item in tsv_:
            sentence = " ".join(token["FORM"] for token in item.words)
            sentences.append(sentence)
        return sentences




if __name__ == "__main__":
    corpus_path = "data/FR/train.cupt"
    preprocessing = Preprocessing(corpus_path)
    sentences = preprocessing.sentences
    print(sentences[3])

