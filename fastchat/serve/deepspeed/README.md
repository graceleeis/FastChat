# How to use fastchat with deepspeed
## use zero-offloading
python3 -m fastchat.serve.cli --model-path facebook/opt-30b --deepspeed --offload

## use deepspeed inference
 python3 -m fastchat.serve.cli --model-path bigscience/bloom-7b1 --deepspeed

# When to use deepspeed
When model is big enough for single GPU, which cause OOM

enable Deepspeed can utilize CPU memory and free you from OOM problem