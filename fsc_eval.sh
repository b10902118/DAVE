export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=3

python3 ./main.py \
--skip_train \
--data_path /project/g/r13922043/dave_dataset/FSC147 \
--model_path /project/g/r13922043/dave_model \
--model_name DAVE_3_shot \
--backbone resnet50 \
--swav_backbone \
--reduction 8 \
--num_enc_layers 3 \
--num_dec_layers 3 \
--kernel_dim 3 \
--emb_dim 256 \
--num_objects 3 \
--num_workers 8 \
--use_query_pos_emb \
--use_objectness \
--use_appearance \
--batch_size 1 \
--pre_norm
