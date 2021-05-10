#!/bin/bash

#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 9 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:8
##SBATCH --gres=gpu:v100_32:1
##SBATCH --gres=gpu:v100_32_maxq:1
#SBATCH --nodelist=ace
##SBATCH --exclude=
#SBATCH -t 7-00:00
##SBATCH -o slurm.%N.%j.out # STDOUT
##SBATCH -e slurm.%N.%j.err # STDERR

pwd
hostname
date

source ~/.bashrc
cd /data/yu_gai/numnet
conda activate numnet

export PYTHONUNBUFFERED=1

export ALLENNLP_CACHE_ROOT=/data/yu_gai/.allennlp

set -xe

# SEED=$1
# LR=$2
# BLR=$3
# WD=$4
# BWD=$5
# TMSPAN=$6

SEED=345
LR=5e-4
BLR=1.5e-5
WD=5e-5
BWD=0.01
TMSPAN=tag_mspan

BASE_DIR=.

DATA_DIR=${BASE_DIR}/drop_dataset
CODE_DIR=${BASE_DIR}

if [ ${TMSPAN} == tag_mspan ];then
  echo "Use tag_mspan model..."
  CACHED_TRAIN=${DATA_DIR}/tmspan_cached_roberta_train.pkl
  CACHED_DEV=${DATA_DIR}/tmspan_cached_roberta_dev.pkl
  MODEL_CONFIG="--gcn_steps 3 --use_gcn --tag_mspan"
  if [ \( ! -e "${CACHED_TRAIN}" \)  -o \( ! -e "${CACHED_DEV}" \) ]; then
  echo "Preparing cached data."
  python prepare_roberta_data.py --input_path ${DATA_DIR} --output_dir ${DATA_DIR} --tag_mspan
  fi
else
  echo "Use mspan model..."
  CACHED_TRAIN=${DATA_DIR}/cached_roberta_train.pkl
  CACHED_DEV=${DATA_DIR}/cached_roberta_dev.pkl
  MODEL_CONFIG="--gcn_steps 3 --use_gcn"
  if [ \( ! -e "${CACHED_TRAIN}" \)  -o \( ! -e "${CACHED_DEV}" \) ]; then
  echo "Preparing cached data."
  python prepare_roberta_data.py --input_path ${DATA_DIR} --output_dir ${DATA_DIR}
  fi
fi


MODEL_ARGS="
  --model_name_or_path ${DATA_DIR}/roberta.large \
  --dropout 0.1 \
  --use_gcn \
  --gcn_steps 1
"
DATA_ARGS="--data_dir ${DATA_DIR}"
# DATA_ARGS="--data_dir ${DATA_DIR} --max_train_samples 64"
OUTPUT_DIR=${BASE_DIR}/numnet_plus_${SEED}_LR_${LR}_BLR_${BLR}_WD_${WD}_BWD_${BWD}${TMSPAN}
TRAINING_ARGS="
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    `# --do_predict` \
    --evaluation_strategy epoch \
    `# --prediction_loss_only` \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --learning_rate ${LR} \
    --weight_decay ${WD} \
    `# --adam_beta1` \
    `# --adam_beta2` \
    `# --adam_epsilon` \
    `# --max_grad_norm` \
    --num_train_epoch 5 \
    `# --max_steps` \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    `# --warmup_steps` \
    --logging_dir ${OUTPUT_DIR} \
    --logging_strategy epoch \
    --logging_first_step \
    `# --logging_steps` \
    --save_strategy epoch \
    `# --save_steps` \
    `# --save_total_limit` \
    `# --no_cuda` \
    --seed ${SEED} \
    `# --fp16` \
    `# --fp16_opt_level` \
    `# --fp16_backend` \
    `# --fp16_full_eval` \
    `# --local_rank` \
    `# --tpu_num_cores` \
    `# --tpu_metrics_debug` \
    `# --debug` \
    --dataloader_drop_last \
    `# --eval_steps` \
    --dataloader_num_workers 16 \
    `# --past_index` \
    `# --run_name` \
    `# --disable_tqdm` \
    `# --remove_unused_columns` \
    `# --label_names` \
    --load_best_model_at_end \
    --metric_for_best_model em \
    --greater_is_better True \
    `# --ignore_data_skip` \
    `# --sharded_ddp` \
    `# --deepspeed` \
    `# --label_smoothing_factor` \
    `# --adafactor` \
    `# --group_by_length` \
    `# --length_column_name` \
    `# --report_to` \
    `# --ddp_find_unused_parameters` \
    --dataloader_pin_memory \
    `# --skip_memory_metrics` \
    `# --use_legacy_prediction_loop` \
    `# --push_to_hub` \
    `# --resume_from_checkpoint` \
    `# --mp_parameters`
"

echo "Start training..."
python -m torch.distributed.launch --nproc_per_node 8 ${CODE_DIR}/roberta_gcn_cli.py \
    ${MODEL_ARGS} \
    ${DATA_ARGS} \
    ${TRAINING_ARGS}

echo "Starting evaluation..."
TEST_CONFIG="--eval_batch_size 5 --pre_path ${OUTPUT_DIR}/checkpoint_best.pt --data_mode dev --dump_path ${OUTPUT_DIR}/dev.json \
             --inf_path ${DATA_DIR}/drop_dataset_dev.json"

python ${CODE_DIR}/roberta_predict.py \
    ${DATA_ARGS} \
    ${TEST_CONFIG} \
    ${MODEL_ARGS} \
    ${MODEL_CONFIG}

python ${CODE_DIR}/drop_eval.py \
    --gold_path ${DATA_DIR}/drop_dataset_dev.json \
    --prediction_path ${OUTPUT_DIR}/dev.json
