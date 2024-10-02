#!/bin/bash

#SBATCH -A STF218
#SBATCH -J gpt22b_megatron
#SBATCH -o logs/gpt22b_megatron_2N_1x8_fa-%j.o
#SBATCH -e logs/gpt22b_megatron_2N_1x8_fa-%j.e
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -N 2

set +x
source /lustre/orion/world-shared/stf218/sajal/miniconda3-frontier/bin/activate
conda activate /lustre/orion/stf006/world-shared/irl1/flash2
#conda activate /lustre/orion/world-shared/stf218/sajal/flash2-copy
module load rocm/5.7.0


# For the plugin
module load libtool
module swap PrgEnv-cray PrgEnv-gnu
module load rocm/$rocm_version
module load craype-accel-amd-gfx90a
module load gcc/12.2.0
module load cray-mpich/8.1.27

export ROCM_HOME=/opt/rocm-5.7.0

export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"
module load PrgEnv-gnu

export ROCM_HOME=/opt/rocm-5.7.0
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
#export NCCL_DEBUG=INFO
# export settings
export TORCH_EXTENSIONS_DIR=$PWD/deepspeed
export HF_HOME=$PWD/hfdata
export OMP_NUM_THREADS=1

# setup hostfile
HOSTS=.hosts-job$SLURM_JOB_ID
HOSTFILE=hostfile.txt
srun hostname > $HOSTS
sed 's/$/ slots=8/' $HOSTS > $HOSTFILE

# setup env file
echo "PATH=$PATH" > .deepspeed_env
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> .deepspeed_env
echo "CPATH=$CPATH" >> .deepspeed_env
echo "TORCH_EXTENSIONS_DIR=$PWD/deepspeed" >> .deepspeed_env
echo "HF_HOME=$PWD/hfdata" >> .deepspeed_env
echo "ROCM_HOME=/opt/rocm-5.7.0" >> .deepspeed_env



scontrol show hostnames $SLURM_NODELIST > job.node.list
input="./job.node.list"
readarray -t arr <"$input"
first=${arr[0]}
echo "first=" $first
ips=`ssh $first hostname -I`
read -ra arr <<< ${ips}
export MASTER_ADDR=${arr[0]}
echo "MASTER_ADDR=" $MASTER_ADDR

ranks_per_node=8
gpus_per_rank=$((8/$ranks_per_node))
ranks_total=$(($ranks_per_node*$SLURM_JOB_NUM_NODES))

echo $ranks_per_node $gpus_per_rank $ranks_total
mkdir logs
mkdir logs/transformer

export CHECKPOINT_PATH=checkpoints/gpt2_345m
export VOCAB_FILE=gpt2-vocab.json
export MERGE_FILE=gpt2-merges.txt
export DATA_PATH=/lustre/orion/world-shared/stf218/sajal/mtds/gptdata/gpttext_article_document

export GPT_ARGS="--tensor-model-parallel-size 1 \
          --pipeline-model-parallel-size 8 \
          --num-layers 48 \
          --hidden-size 6144 \
          --num-attention-heads 48 \
          --seq-length 2048 \
          --max-position-embeddings 2048 \
  	  --micro-batch-size 1 \
	  --global-batch-size 640 \
	  --lr 6.0e-5 \
	  --train-iters 100 \
          --lr-decay-iters 10 \
          --lr-decay-style cosine \
	  --min-lr 6.0e-6 \
          --lr-decay-style cosine \
          --log-interval 1 \
          --eval-iters 3 \
          --eval-interval 3 \
	  --vocab-file $VOCAB_FILE \
	  --merge-file $MERGE_FILE \
          --split 98,2,0 \
          --clip-grad 1.0 \
	  --weight-decay 0.1 \
	  --adam-beta1 0.9 \
	  --adam-beta2 0.95 \
	  --init-method-std 0.006 \
          --fp16"
#--train-samples 1000 \
#       	  --lr-decay-samples 100 \
#          --lr-warmup-samples 100 \

export OUTPUT_ARGS="--log-interval 1 \
             --eval-interval 1 \
             --eval-iters 2"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export LD_LIBRARY_PATH=opt/cray/libfabric/1.15.2.0/lib64:/lustre/orion/world-shared/stf218/sajal/rccl-plugin/aws-ofi-rccl/src/.libs/:/opt/rocm-5.7.0/lib:${LD_LIBRARY_PATH}
#export NCCL_NET_GDR_LEVEL=3
#export FI_MR_CACHE_MONITOR=userfaultfd

time srun -u -n$ranks_total -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest python pretrain_gpt_deepspeed.py \
       $GPT_ARGS \
       --master-addr=$MASTER_ADDR \
       --data-path $DATA_PATH \
       --num-workers 0 \
       --tensorboard-dir logs/profiles-with-zero-0 \
       --deepspeed \
       --deepspeed_config ds_config_16N_175B.json \
       --deepspeed-activation-checkpointing \
       --checkpoint-activations \
       --zero-stage 0 \
       --use-flash-attn \
       --use-flash-attn-v2
