from bert_embedding import BertEmbedding

import sys
sys.path.append(".")
from preprocessing import Preprocessing

class BertWordEmbedding:
    def __init__(self, sentence):
        self.sentences = sentences
        self.model = "bert_12_768_12"
        self.dataset_name = "wiki_multilingual_cased"

    def vectorize(self):
        bert = BertEmbedding(model=self.model, dataset_name=self.dataset_name)
        vector = bert(self.sentences)
        return vector


if __name__ == "__main__":
    corpus_path = "data/FR/train.cupt"
    prepro = Preprocessing(corpus_path)
    sentences = prepro.sentences
    bert = BertWordEmbedding(sentences)
    vector = bert.vectorize()
    print(vector)
    print(type(vector))
    print(len(vector))
    print(vector[0][0])
    print(vector[0][1])
    print(type(vector[0][0]))
