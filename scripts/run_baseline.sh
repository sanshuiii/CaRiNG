#/bin/bash
# UG
python train_stationary.py \
    --exp tdrl_ng \
    --seed 770


# UG-TDMP
python train_stationary.py \
    --exp tdrl_ng_tdmp \
    --seed 770
