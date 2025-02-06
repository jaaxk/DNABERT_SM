#!/bin/bash
#SBATCH --job-name=my_job         # Job name
#SBATCH --partition=a100-long      # Partition name
#SBATCH --time=48:00:00            # Max runtime (48 hours)
#SBATCH --output=res_ddp.txt           # Standard output and error log
#SBATCH --gres=gpu:4         # Request 4 GPUs
#SBATCH --mem=250G

source /gpfs/scratch/jvaska/DNABERT_SM/myenv/bin/activate

export PATH_TO_DATA_DICT=/gpfs/scratch/jvaska/DNABERT_SM/train/pretrain/data
export TRAIN_FILE=train_2m.csv
export HF_HOME=/gpfs/scratch/jvaska/DNABERT_SM/huggingface_cache
export NUM_GPUS=4
export OMP_NUM_THREADS=$NUM_GPUS
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"

module load pytorch-gpu

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    main.py \
    --resdir ./results/ \
    --datapath ${PATH_TO_DATA_DICT} \
    --train_dataname ${TRAIN_FILE} \
    --val_dataname val_48k.csv \
    --seed 1 \
    --logging_step 10000 \
    --logging_num 12 \
    --max_length 2000 \
    --train_batch_size 6 \
    --val_batch_size 45 \
    --lr 3e-06 \
    --lr_scale 100 \
    --epochs 3 \
    --feat_dim 128 \
    --temperature 0.05 \
    --con_method same_species \
    --mix \
    --mix_alpha 1.0 \
    --mix_layer_num -1 \
    --curriculum
