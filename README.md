# Is Large Language Model All You Need for 3D Scene Understanding

Ruoyu Wang, Weihan Xu, Xihang Yu

# Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Running Code](#running-code)

## Overview

Robotic applications rely on scene understanding to analyze objects within a 3D environment. One crucial component of scene understanding is semantics labeling, which involves assigning class labels to semantic regions based on the objects within them. In a recent study [Leveraging Large Language Model for 3D Scene Understanding](https://arxiv.org/abs/2209.05629) , Large Language Models (LLMs) were found to be effective in incorporating common sense knowledge during the labeling process. In this project, we aim to compare two LLMs, GPT-J and RoBERTa, using fine-tuned feed-forward and contrastive networks, which were not evaluated in \cite{chen2022leveraging}, for the semantic labeling task. The contributions of this project are twofold: (i) The proposed GPT-J with fine-tuned feed-forward network achieves state-of-the-art(SOTA) performance, and (ii) by varying the number of candidate objects, adopting ChatGPT-based room detection and fine-tuning a whole BERT-based network, we explore the possible performance bottleneck of our proposed GPT-J pretrained network.

## Requirements
Before starting, you will need:
- A CUDA-enabled GPU (we used an RTX 3080 with 16 GB of memory)
- A corresponding version of CUDA (we used v11.1)
- Python 3.8 with venv
- Pip package manager

After cloning this repo: 
- Create a virtual environment `python3 -m venv /path/to/llm_su_venv`
- Source the environment `source /path/to/llm_su_venv/bin/activate`
- Enter this repo and install all requirements: `pip install -r requirements.txt`
  - Note that some libraries listed in that file are no longer necessary, as it was procedurally generated. One can alternatively go through the scripts one wishes to run and install their individual dependencies.
  - Such dependencies include: `numpy, scipy, torch, torch_geometric, torchvision, matplotlib, transformers, tqdm, pandas, gensim, sympy`.

## Running Code
- `python zero_shot_<rooms/bldgs>.py` runs our zero-shot language approach on the entire Matterport3D dataset to predict either rooms given objects or buildings given rooms.
- `python <ff/contrastive>_train(_gptj).py` runs our feed-forward or contrastive training approaches respectively.
  - Run `python data_generator(_gptj).py` prior to the above to generate the bootstrapped data needed for training and evaluation.
- `python bldg_ff_train(_gptj).py` and `python bldg_data_generator_comparison(_gptj).py` are the equivalents for building-prediction. Note that said data generator does _not_ bootstrap datapoints for the test set, instead just using the same test set as `zero_shot_bldgs.py` for easier comparison.
- `python create_label_embedding_gptj.py` to create room label strings embeddings for contrastive network.
- `python <ff/contrastive>_holdout_tests.py` runs training on a dataset with certain objects withheld, then evaluating on datapoints with those previously-unseen objects.
- `python <ff/contrastive>_label_space_test.py` runs training on the mpcat40 label space dataset, then evaluates on the larger nyuClass label space dataset.
- Some other utility functions and scripts are included as well, such as `compute_cooccurrencies.py`, which generates co-occurrency matrices (i.e. counting frequencies of room-object pairs)
