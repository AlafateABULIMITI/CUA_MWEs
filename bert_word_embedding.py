from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer


class BertWordEmbedding:
    def __init__(self):
        self.port = port
        self.model_dir = model_dir
        self.port_out = port_out
        self.max_seq_len = max_seq_len
        self.tuned_model_dir = tuned_model_dir
        self.mask_cls_sep = (
            mask_cls_sep
        )  # masking the embedding on [CLS] and [SEP] with zero.
        self.pooling_strategy = (
            pooling_strategy
        )  # the pooling strategy for generating encoding vectors, valid values are NONE, REDUCE_MEAN, REDUCE_MAX, REDUCE_MEAN_MAX, CLS_TOKEN, FIRST_TOKEN, SEP_TOKEN, LAST_TOKEN. Explanation of these strategies can be found https://github.com/hanxiao/bert-as-service#q-what-are-the-available-pooling-strategies. To get encoding for each token in the sequence, please set this to NONE.


args = get_args_parser().parse_args(
    [
        "-model_dir",
        "YOUR_MODEL_PATH_HERE",
        "-port",
        "5555",
        "-port_out",
        "5556",
        "-max_seq_len",
        "NONE",
        "-mask_cls_sep",
        "-cpu",
    ]
)
server = BertServer(args)
server.start()
