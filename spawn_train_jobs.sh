#!/bin/bash

nohup python train.py exps_hyperparam_tuning/double_lr_half_num_epoch/trial_4/config.yaml > exps_hyperparam_tuning/double_lr_half_num_epoch/trial_4/run_output.log &
nohup python train.py exps_hyperparam_tuning/double_lr_half_num_epoch/trial_5/config.yaml > exps_hyperparam_tuning/double_lr_half_num_epoch/trial_5/run_output.log &
nohup python train.py exps_hyperparam_tuning/higher_weight_decay/trial_4/config.yaml > exps_hyperparam_tuning/higher_weight_decay/trial_4/run_output.log &
nohup python train.py exps_hyperparam_tuning/higher_weight_decay/trial_5/config.yaml > exps_hyperparam_tuning/higher_weight_decay/trial_5/run_output.log &
# python train.py exps_hyperparam_tuning/small_batch_size/config.yaml > exps_hyperparam_tuning/small_batch_size/run_output.log &

