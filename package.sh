#!/bin/bash
conda create -n env_mwei python=3.6
conda activate env_mwei
conda install -c conda-forge matplotlib
conda install -c conda-forge tensorflow=1.10.0
conda install pytorch torchvision -c pytorch
conda install -c pytorch torchtext
pip install bert-serving-server # server
pip install bert-serving-client # client, independent of `bert-serving-server`
pip install conllu
