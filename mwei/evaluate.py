import sys
from shutil import copyfile

sys.path.append("./utils")
import tsvlib
from conllu import parse_incr


class Evaluation:
    def __init__(self, eval_file, out_file):
        super().__init__()
        self.eval_file = eval_file
        self.out_file = out_file

    def _copy(self):
        copyfile(self.eval_file, out_file)

    def evaluate(self, model):
        pass

    
