#!/usr/bin/env bash

EXP_NAME=$1
SAVE_DIR=checkpoints/${EXP_NAME}
IMAGENET_PRETRAIN=/data/.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=/data/.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path
SPLIT_ID=$2
NUNMGPU=2
URL=8186

BASE_OUTPUT_DIR = ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
# ------------------------------- Base Pre-train ---------------------------------- #
# sub-stage1
python3 train.py --num-gpus ${NUNMGPU} --dist-url tcp://127.0.0.1:4${URL}\
    --config-file configs/qiunao/defrcn_det_r101_base${SPLIT_ID}.yaml     \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
           OUTPUT_DIR ${BASE_OUTPUT_DIR}

# sub-stage2
BASE_WEIGHTS = ${BASE_OUTPUT_DIR}/model_final.pth
BASE_GCN_OUTPUT_DIR = ${SAVE_DIR}/base${SPLIT_ID}_gcn

python3 train.py --num-gpus ${NUNMGPU} --dist-url tcp://127.0.0.1:4${URL} \
    --config-file configs/qiunao/base_gcn/defrcn_det_r101_base${SPLIT_ID}.yaml \
    --opts MODEL.WEIGHTS ${BASE_WEIGHTS} OUTPUT_DIR ${BASE_GCN_OUTPUT_DIR}

# ------------------------------ Model Preparation -------------------------------- #
python3 tools/model_surgery.py --dataset qiunao --method remove                                    \
    --src-path ${BASE_GCN_OUTPUT_DIR}/model_final.pth \
    --save-dir ${BASE_GCN_OUTPUT_DIR}
# ------------------------------ Novel Fine-tuning -------------------------------- #
BASE_GCN_WEIGHT = ${BASE_GCN_OUTPUT_DIR}/model_reset_surgery.pth
FINETUNE_OUTPUT_DIR = ${SAVE_DIR}/finetune${SPLIT_ID}
for shot in 1 2 3 5 10   # if final, 10 -> 1 2 3 5 10
do
    for seed in 0
    do
        # sub-stage1
        CONFIG_ROOT = configs/qiunao/finetune_s1
        CONFIG_PATH = ${CONFIG_ROOT}/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        S1_OUTPUT_DIR = ${FINETUNE_OUTPUT_DIR}/${shot}shot_seed${seed}/s1
        python3 tools/create_config.py --dataset qiunao --config_root ${CONFIG_ROOT} \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}

        python3 train.py --num-gpus ${NUNMGPU} --dist-url tcp://127.0.0.1:4${URL}  \
            --config-file ${CONFIG_PATH} \
            --opts MODEL.WEIGHTS ${BASE_GCN_WEIGHT} OUTPUT_DIR ${S1_OUTPUT_DIR}

        # sub-stage2
        FINE_TUNE_WEIGHT = ${S1_OUTPUT_DIR}/model_best_AP50.pth
        S2_OUTPUT_DIR = ${FINETUNE_OUTPUT_DIR}/${shot}shot_seed${seed}/s2
        CONFIG_ROOT = configs/${DATASET}/finetune_s2
        CONFIG_PATH = ${CONFIG_ROOT}/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        python3 tools/create_config.py --dataset ${DATASET} --config_root ${CONFIG_ROOT} \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}

        python3 train.py --num-gpus ${NUNMGPU} --dist-url tcp://127.0.0.1:4${URL} \
            --config-file ${CONFIG_PATH}  \
            --opts MODEL.WEIGHTS ${FINE_TUNE_WEIGHT} OUTPUT_DIR ${S2_OUTPUT_DIR} ${special_config}

        rm ${CONFIG_PATH}
    done
done