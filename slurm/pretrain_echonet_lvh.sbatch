#!/bin/bash
# SSL pretraining (SimCLR + frame re-ordering) on EchoNet-LVH.
#
# Submit from the repository root (the logs/ directory must exist first):
#   mkdir -p logs
#   sbatch slurm/pretrain_echonet_lvh.sbatch
#
# Override settings at submission time, e.g.:
#   sbatch --export=ALL,BATCH_SIZE=128,NUM_EPOCHS=100 slurm/pretrain_echonet_lvh.sbatch
#   sbatch --gres=gpu:2 slurm/pretrain_echonet_lvh.sbatch
#
# Prerequisites:
#   - `uv sync --frozen` run once on a login node (compute nodes often lack internet)
#   - Full EchoNet-LVH release under $DATA_DIR (Batch*/ folders + MeasurementsList.csv)
#   - ssl_pretraining/main.py supports --dataset echonet_lvh

#SBATCH --job-name=ssl-echonet-lvh
#SBATCH --partition=gpu                 # cluster-specific: set to your GPU partition
##SBATCH --account=your_account         # cluster-specific: remove one '#' to enable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16              # main.py uses 12 DataLoader workers + main process
#SBATCH --mem=64G                       # each worker decodes a full 768x1024 video (~350 MB)
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue                    # main.py cannot resume; a requeued run would restart from epoch 1

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/runs}"
CLIP_LEN="${CLIP_LEN:-4}"
SAMPLING_RATE="${SAMPLING_RATE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"          # per GPU; the DataLoader uses N_GPU * BATCH_SIZE
NUM_EPOCHS="${NUM_EPOCHS:-300}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEMPERATURE="${TEMPERATURE:-0.05}"
LR="${LR:-0.1}"
# Job ID in the name: main.py deletes an existing run folder with the same name.
MODEL_NAME="${MODEL_NAME:-echonet_lvh_clip${CLIP_LEN}_sr${SAMPLING_RATE}_${SLURM_JOB_ID}}"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Repo: ${REPO_DIR} | Data: ${DATA_DIR} | Output: ${OUTPUT_DIR}/${MODEL_NAME}"

# ---- Pre-flight checks -------------------------------------------------------
if [[ ! -f "${DATA_DIR}/MeasurementsList.csv" ]]; then
    echo "ERROR: ${DATA_DIR}/MeasurementsList.csv not found" >&2
    exit 1
fi

if [[ -e "${OUTPUT_DIR}/${MODEL_NAME}" ]]; then
    echo "ERROR: ${OUTPUT_DIR}/${MODEL_NAME} exists and main.py would delete it; choose another MODEL_NAME" >&2
    exit 1
fi

if ! grep -q -- "--dataset" "${REPO_DIR}/ssl_pretraining/main.py"; then
    echo "ERROR: ssl_pretraining/main.py has no --dataset option yet" >&2
    exit 1
fi

# ---- Environment -------------------------------------------------------------
# PyPI torch wheels bundle the CUDA runtime, so only the NVIDIA driver is needed.
# If your cluster requires it, load modules here (e.g., `module load cuda`).
source "${REPO_DIR}/.venv/bin/activate"
export PYTHONUNBUFFERED=1               # flush prints to the log immediately
export TQDM_MININTERVAL=60              # limit progress-bar updates in the log file

nvidia-smi

# Count GPUs visible to PyTorch (respects CUDA_VISIBLE_DEVICES set by Slurm)
N_GPU="${N_GPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
python -c 'import torch; print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, available={torch.cuda.is_available()}")'
if [[ "${N_GPU}" -lt 1 ]]; then
    echo "ERROR: PyTorch sees no GPUs" >&2
    exit 1
fi
echo "Using ${N_GPU} GPU(s), per-GPU batch size ${BATCH_SIZE}"

# ---- Train -------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}/ssl_pretraining"        # main.py uses bare imports (from dataset import ...)

srun python main.py \
    --dataset echonet_lvh \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --n_gpu "${N_GPU}" \
    --batch_size "${BATCH_SIZE}" \
    --clip_len "${CLIP_LEN}" \
    --sampling_rate "${SAMPLING_RATE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --save_freq "${SAVE_FREQ}" \
    --temperature "${TEMPERATURE}" \
    --lr "${LR}"

echo "Finished at $(date)"
