#! /usr/bin/env python3

import argparse
import re
import sys

parser = argparse.ArgumentParser(description="""
        Convert check if file is in the CUPT format (edition 1.1).""")
parser.add_argument("--underspecified-mwes", action='store_true',
        help="""If set, check that all MWEs are underspecified as "_" (for blind).""")
parser.add_argument("--input", type=argparse.FileType('r'), required=True,
        help="""Path to input file (in FoLiA XML or PARSEME TSV format)""")


VALID_CATEGS = {'VID', 'LVC.full', 'LVC.cause', 'IRV', 'VPC.full', 'VPC.semi', 'MVC', 'IAV'}


class Main:
    def __init__(self, args):
        self.args = args

    def run(self):
        # NOTE: This code is ugly, and should be reworked to use tsvlib
        # in the future... For now, we'll keep it as it is...

        # TODO: We should validate that two MWEs do not ever have the same
        # set of tokens. Even if two MWEs have different category, if they
        # have the same tokens, we should warn.

        with self.args.input as f:
            header = next(f)
            if "global.columns =" not in header:
                exit("ERROR: first line must specify global.columns")
            colnames = header.split("=")[-1].strip().split()
            try:
                parseme_colname = colnames.index("PARSEME:MWE")
            except ValueError:
                exit("ERROR: missing column PARSEME:MWE")

            mweid2categ = {}
            for lineno, line in enumerate(f, 2):
                if not line.strip() or line.startswith('#'):
                    mweid2categ = {}
                else:
                    fields = line.strip().split("\t")
                    if len(fields) != len(colnames):
                        exit(
                            f"ERROR: line {lineno} has only {len(fields)} columns (expected {len(colnames)})"
                        )

                    token = dict(zip(colnames, fields))
                    for k, v in token.items():
                        if v == "":
                            exit(f"ERROR: line {lineno} has empty {k} field")
                        if k == "PARSEME:MWE" and self.args.underspecified_mwes and v != "_":
                            exit(
                                f'ERROR: line {lineno} has a specific PARSEME:MWE value (expecting the underspecified "_")'
                            )
                        if k == "PARSEME:MWE" and not self.args.underspecified_mwes and v == "_":
                            exit(
                                f"ERROR: line {lineno} has an unexpected PARSEME:MWE value (if this is blind data, use --underspecified)"
                            )

                        if k == "PARSEME:MWE" and v not in "*_":
                            for mwe in v.split(";"):
                                try:
                                    mweid = int(mwe)
                                except ValueError:
                                    try:
                                        mweid, mwecateg = mwe.split(':')
                                        mweid = int(mweid)
                                    except ValueError:
                                        exit("ERROR: line {} has MWE code {!r} (expecting " \
                                                 "an integer like '3' a pair like '3:LVC.full')" \
                                                 .format(lineno, mwe))
                                    else:
                                        if mweid in mweid2categ:
                                            exit(
                                                f"ERROR: line {lineno} redefines a category ('{mweid}:{mweid2categ[mweid]}' => '{mweid}:{mwecateg}')"
                                            )
                                        if mwecateg not in VALID_CATEGS and not mwecateg.startswith('LS.'):
                                            exit(f"ERROR: line {lineno} refers to an invalid category name ('{mwecateg}')")
                                        mweid2categ[mweid] = mwecateg
                                else:
                                    if mweid not in mweid2categ:
                                        exit(
                                            f"ERROR: line {lineno} refers to MWE '{mweid}' without giving it a category right away"
                                        )
            print("Validated: no errors.", file=sys.stderr)



#####################################################

if __name__ == "__main__":
    Main(parser.parse_args()).run()
