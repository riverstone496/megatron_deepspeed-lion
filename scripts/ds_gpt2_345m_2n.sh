source .env/bin/activate
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + (${JOB_ID} % 50000)))
echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
NODE_TYPE="h100"

# GPU resources
NUM_NODES=$NHOSTS
NUM_GPUS_PER_NODE=4
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))
echo "NUM_NODES=${NUM_NODES}"
echo "NUM_GPUS=${NUM_GPUS}"

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
    echo "${hostname} slots=${NUM_GPUS_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

# Dataset path & checkpoint path
DATASET_PATH=dataset/arxiv_text_document
CHECKPOINT_PATH=checkpoints/gpt2_345m/ds_8gpu
mkdir -p ${CHECKPOINT_PATH}

VOCAB_PATH=dataset/gpt2-vocab.json
MERGE_PATH=dataset/gpt2-merges.txt

# GPT-2 345M (24-layer, 1024-hidden, 16-heads, 345M parameters)
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16

# Parellel parameters
PP_SIZE=1
TP_SIZE=1

DP_SIZE=$((${NUM_GPUS} / (${PP_SIZE} * ${TP_SIZE})))

# Training parameters
GRAD_ACCUMULATION_STEPS=1

MICRO_BATCHSIZE=8
GLOBAL_BATCH_SIZE=$((MICRO_BATCHSIZE * DP_SIZE))

SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=1024

TRAINING_ITERATIONS=500000
SAVE_INTERVAL=10000
LR_DECAY_ITERATIONS=320000

LR=0.00015
LR_WARMUP_ITER=32000
SEED=1234

# deepspeed configuration
CONFIG_FILE=scripts/ds_config_gpt2_345m_${NUM_GPUS}.json
ZERO_STAGE=1

export NCCL_DEBUG=WARN

# Run Command
mpirun -np ${NUM_GPUS} \
  --npernode ${NUM_GPUS_PER_NODE} \
  -hostfile ${HOSTFILE_NAME} \
  -x MASTER_ADDR=${MASTER_NODE} \
  -x MASTER_PORT=${MASTER_PORT} \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x LD_LIBRARY_PATH \
  -x PATH \
  -bind-to none \
  -x PATH \
  python pretrain_gpt.py \
  --tensor-model-parallel-size ${TP_SIZE} \
  --pipeline-model-parallel-size ${PP_SIZE} \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --num-attention-heads ${NUM_ATTN_HEADS} \
  --micro-batch-size ${MICRO_BATCHSIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
  --train-iters ${TRAINING_ITERATIONS} \
  --save-interval ${SAVE_INTERVAL} \
  --lr-decay-iters ${LR_DECAY_ITERATIONS} \
  --data-path ${DATASET_PATH} \
  --vocab-file ${VOCAB_PATH} \
  --merge-file ${MERGE_PATH} \
  --data-impl mmap \
  --split 949,50,1 \
  --save ${CHECKPOINT_PATH} \
  --load ${CHECKPOINT_PATH} \
  --distributed-backend nccl \
  --override-lr-scheduler \
  --lr $LR \
  --lr-decay-style cosine \
  --min-lr 1.0e-5 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-iters $LR_WARMUP_ITER \
  --checkpoint-activations \
  --log-interval 100 \
  --eval-interval 100 \
  --eval-iters 10 \
  --fp16 \
  --seed $SEED \
  --no-masked-softmax-fusion \
  --deepspeed \
  --deepspeed_config ${CONFIG_FILE} \
  --zero-stage ${ZERO_STAGE} \
  --deepspeed-activation-checkpointing \
  --optimizer mvlion \
