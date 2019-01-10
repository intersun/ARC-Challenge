# ARC-Challenge Baseline
This repo describes how to get the baseline score for ARC challenge dataset using BERT

The major part of code is obtained from https://github.com/huggingface/pytorch-pretrained-BERT, and run_arc.py is modified based on run_swag.py in the repo. 

## Pre-trained on RACE
The first step is to pre-train the model on RACE (**both** M and H), and select epoch and other hyper parameters (step size and batch size) by a **merged** RACE validation dataset. After this step, the BERT-large model should have ~67% (accuracy) on RACE-H test and ~40% on ARC-Challenge test, and BERT-base should have ~63% on RACE-H test and ~36% on ARC-Challenge test.

## Finetune on ARC
In this step, you need to train on **merged** ARC-Easy and ARC-challenge based on the previous pre-trained model, and select hyper-parameters based on the **merged** ARC-Easy and ARC-Challenge validation dataset (selected model has 58% or so accuracy on this merged validation data). After this step, the single model performance on ARC-Challenge test is around 44-45% and 68-69% on ARC-Easy. Performance could be further boosted by ensembling more models. 

## Dataset
Race dataset could be obtained following this link https://github.com/qizhex/RACE_AR_baselines. 
For ARC dataset, we are using the method in https://arxiv.org/abs/1808.09492 to prepare passages.
Also please refer to the data folder for how to place data and pretrained model to make the code work, it is organized this way mainly for training on Microsoft server purpose.

