checkpoint="Qwen-Audio"
data_config_path="data_config.jsonl"
output_dir="/user/likehan/output_qwen_audio"
mkdir -p $output_dir

NPROC_PER_NODE=4 # Set the number of GPUs per node
NODES=1 # Set the number of nodes

python -m torch.distributed.launch --use-env \
    --nnodes=$NODES \
    --nproc_per_node=$NPROC_PER_NODE \
    SER.py \
    --checkpoint $checkpoint \
    --data_config_path $data_config_path \
    --output_dir $output_dir \
    --batch-size 8 \
    --num-workers 2