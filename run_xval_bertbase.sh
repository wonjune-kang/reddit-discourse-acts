#!/usr/bin/env bash

for i in {0..9}; do
    python3 -u train.py \
        --run_name bertbase_subtree_3_wd_0.0 \
        --data_path ../data/reddit_coarse_discourse_clean \
        --num_gpus 4 \
        --num_epochs 10 \
        --lr 2e-5 \
        --batch_size 32 \
        --weight_decay 0.0 \
        --bert_encoder_type BERT-Base \
        --dim_feedforward 768 \
        --dropout 0.1 \
        --max_subtree_depth 3 \
        --use_ancestor_labels \
        --randomize_prob 0.1 \
        --xval_test_idx $i \
        --save_path ./results
done
