from bert_serving.server.helper import get_args_parser
from bert_serving.server.helper import get_shutdown_parser
from bert_serving.server import BertServer
from bert_serving.client import BertClient
import numpy as np

START_ARGS = [
    "-model_dir",
    "../models/multi_cased_L-12_H-768_A-12",
    "-num_worker",
    "2",
    "-port",
    "5555",
    "-port_out",
    "5556",
    "-max_seq_len",
    "NONE",
    "-pooling_strategy",
    "NONE",
    "-mask_cls_sep",
    "-cpu",
]
SHUT_ARGS = ["-ip", "localhost", "-port", "5555", "-timeout", "5000"]


class BertWordEmbedding:
    def __init__(self):
        self.start_args = get_args_parser().parse_args(START_ARGS)
        self.shut_args = get_shutdown_parser().parse_args(SHUT_ARGS)

    def vectorize(self, client, tokens):
        return np.squeeze(client.encode(tokens, is_tokenized=True))[1:-1]


if __name__ == "__main__":
    bert = BertWordEmbedding()
    tokens = [["hello", "world", "!"]]
    with BertServer(bert.start_args):
        with BertClient() as client:
            vecs = bert.vectorize(client, tokens)
    print(vecs)
    print(vecs.shape)
