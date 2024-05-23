#/bin/bash
# UG
python train_stationary.py \
    --exp tdrl_ug \
    --seed 770


# UG-TDMP
python train_stationary.py \
    --exp tdrl_ug_tdmp \
    --seed 770
