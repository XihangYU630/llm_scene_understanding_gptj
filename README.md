# Is large language model all you need for 3D understanding? 

Ruoyu Wang, Xihang Yu, and Weihan Xu



# Table of Contents
1. [Adjustments](#Adjustments)
2. [Running Code](#running-code)



## Adjustments
This code is adapted from this github repo https://github.com/MIT-SPARK/llm_scene_understanding
We modified the dataset class, the model structure, and the training process to fine-tune the BERT classifier. All our modified python files are names as R_xx.py.

## Running Code
- `python R_generate_data.py` generates data needed 
- `python R_ff_train.py` trains the BERT classifier
- `python R_test.py` analyzes the output and the training process
