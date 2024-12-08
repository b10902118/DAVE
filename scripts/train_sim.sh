#!/bin/bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=3
# 4 -> 2
# 3 -> 1
# 2 -> 0
CUDA_VISIBLE_DEVICES=5 python ./train_similarity.py \
--model_name base_3_shot \
--det_model_name verification \
--data_path /project/g/r13922043/dave_dataset/FSC147 \
--model_path /project/g/r13922043/dave_model/ \
--backbone resnet50 \
--swav_backbone \
--reduction 8 \
--image_size 512 \
--num_enc_layers 3 \
--num_dec_layers 3 \
--kernel_dim 3 \
--emb_dim 256 \
--num_objects 3 \
--epochs 50 \
--lr 1e-5 \
--lr_drop 220 \
--weight_decay 1e-2 \
--batch_size 32 \
--dropout 0.1 \
--num_workers 8 \
--max_grad_norm 0.1 \
--normalized_l2 \
--detection_loss_weight 0.01 \
--count_loss_weight 0.0 \
--min_count_loss_weight 0.0 \
--aux_weight 0.3 \
--tiling_p 0.4 \
--use_query_pos_emb \
--use_objectness \
--use_appearance \
--pre_norm \