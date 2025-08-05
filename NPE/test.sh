#!/bin/bash

QP=8
GPU=1
CLASS_LIST="Class_B Class_C Class_D"
EXP_NAME="npe_for_test"
CHECKPOINT="./checkpoint/checkpoint_best.pth.tar"
OUTPUT_DIR="./npe_out/${EXP_NAME}"

for CLASS in $CLASS_LIST; do
    CUDA_VISIBLE_DEVICES=$GPU python test_yuv_video.py \
        --dataset ./HM_dataset/HM_dataset_classes/$CLASS \
        --output ${OUTPUT_DIR}/${EXP_NAME} \
        --cuda -f \
        -c $CHECKPOINT \
        --only_yuv
done
