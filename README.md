# MWEIs
The code is for indentification of Multiwords Expressions.

## data
1. from the PARSEME website
2. French data for now

## model
1. from the [bert](https://github.com/google-research/bert)
2. BERT-Base, Multilingual Cased  : 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters

### why not just for the french ?
> Here is a comparison of training Chinese models with the Multilingual BERT-Base and Chinese-only BERT-Base:

| System                  | Chinese |
| ----------------------- | :-----: |
| XNLI Baseline           |  67.0   |
| BERT Multilingual Model |  74.2   |
| BERT Chinese-only Model |  77.2   |
> Similar to English, the single-language model does 3% better than the Multilingual model.

for more infos : [bert multilingual](https://github.com/google-research/bert/blob/master/multilingual.md)

