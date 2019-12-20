import sys
from shutil import copyfile

sys.path.append("./utils")
import tsvlib
from conllu import parse_incr
from collections import Counter


class Evaluation:
    def __init__(self, eval_corpus, result):
        super().__init__()
        self.eval_corpus = eval_corpus
        self.result = result

    def _convert(self, vectors):
        mwei_preds_list = list()
        for vec in vectors:
            preds = model.predict(vec)
            mwei_preds_list.self._transform_preds(preds)
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
    def write(self, vectors):
        mwei_preds_list = self._convert(vectors)
        with open(self.eval_corpus, "r", encoding="utf-8") as data_file:
            with open(self.result, "w+", encoding="utf-8") as save_to:
                for tokenlist in parse_incr(data_file):
                    for preds in mwei_preds_list:
                        print(tokenlist)
                        print(tokenlist.metadata)
                        for idx, token in enumerate(tokenlist):
                            if "-" in token["id"]:
                                token["parseme:mwe"] = "*"
                            else:
                                token["parseme:mwe"] = preds[int(token["id"])]
                    a = tokenlist.serialize()
                    save_to.write(a)
                    print(a)


# with open("../tmp/train_test.cupt", "r", encoding="utf-8") as data_file:
#     with open("../tmp/train_test_tr.cupt", "w+", encoding="utf-8") as save_to:
#         for tokenlist in parse_incr(data_file):
#             print(tokenlist)
#             print(tokenlist.metadata)
#             tokenlist[0]["parseme:mwe"] = "111111"
#             token = tokenlist[0]
#             print(token)
#             a = tokenlist.serialize()
#             save_to.write(a)
#             print(a)
#             break
