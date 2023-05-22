export WANDB_DISABLED=true
export NCCL_SOCKET_IFNAME=eth0
USER=xiaoranli
export PROJECT_DIR=/home/${USER}/FastChat
# export model=facebook/opt-30b
export model=bigscience/bloom-7b1

reset

python3 -m fastchat.serve.cli --model-path $model --deepspeed
# cannot use multiple GPU ds in this way: python3 -m fastchat.serve.cli --model-path facebook/opt-30b --deepspeed --num-gpus 2
# python3 -m fastchat.serve.cli --model-path facebook/opt-30b  --cpu-offloading --load-8bit
# python3 -m fastchat.serve.cli --model-path facebook/opt-30b  --cpu-offloading --load-8bit --num-gpus 2
# torchrun --nnodes=1 --nproc_per_node=2  $PROJECT_DIR/fastchat/serve/cli.py --model-path facebook/opt-30b --deepspeed
# also wrong: deepspeed --num_nodes=1 --num_gpus=2  $PROJECT_DIR/fastchat/serve/cli.py --model-path facebook/opt-30b