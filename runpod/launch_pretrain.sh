#!/usr/bin/env bash
# Launch SSL pretraining on a Runpod pod inside tmux, then stop the pod when training ends.
#
# Usage (on the pod, with the repository under /workspace):
#   bash runpod/launch_pretrain.sh
#   BATCH_SIZE=128 NUM_EPOCHS=100 bash runpod/launch_pretrain.sh
#   AUTO_STOP=0 bash runpod/launch_pretrain.sh        # leave the pod running afterwards
#
# Monitor:  tmux attach -t ssl-pretrain   (detach with Ctrl-b d)
#           tail -f /workspace/logs/<model_name>.log
# Cancel:   attach and press Ctrl-C. This ends training and skips the pod stop.
#
# The pod is stopped whether training succeeds or fails, so a crash does not leave
# an idle GPU billing. Stopping keeps /workspace; terminating would delete it.

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

# Pod variables may be missing in SSH or non-interactive shells; some Runpod
# images export them to /etc/rp_environment.
load_runpod_env() {
    if [[ -z "${RUNPOD_POD_ID:-}" || -z "${RUNPOD_API_KEY:-}" ]] && [[ -f /etc/rp_environment ]]; then
        # shellcheck disable=SC1091
        source /etc/rp_environment
    fi
}

stop_pod() {
    load_runpod_env
    if command -v runpodctl >/dev/null 2>&1 \
        && runpodctl stop pod "$RUNPOD_POD_ID" >>"$LOG_FILE" 2>&1; then
        log "Stop requested via runpodctl."
        return 0
    fi
    if [[ -n "${RUNPOD_API_KEY:-}" ]] \
        && curl -fsS -X POST "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" >>"$LOG_FILE" 2>&1; then
        log "Stop requested via REST API."
        return 0
    fi
    log "ERROR: could not stop pod ${RUNPOD_POD_ID}; stop it from the Runpod console."
    return 1
}

run_training() {
    set +e    # keep going after a training failure so the pod still gets stopped
    trap 'log "Interrupted by user; pod will NOT be stopped."; exit 130' INT

    # shellcheck disable=SC1091
    source "${REPO_DIR}/.venv/bin/activate"
    export PYTHONUNBUFFERED=1
    export TQDM_MININTERVAL=30

    log "Starting ${MODEL_NAME} on ${N_GPU} GPU(s); log: ${LOG_FILE}"
    cd "${REPO_DIR}/ssl_pretraining" || exit 1    # main.py uses bare imports

    python main.py \
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
        --lr "${LR}" 2>&1 | tee -a "$LOG_FILE"
    local status=${PIPESTATUS[0]}
    log "Training exited with status ${status}."

    if [[ "${AUTO_STOP}" == "1" ]]; then
        log "Stopping pod in ${STOP_DELAY}s (attach and press Ctrl-C to cancel)."
        sleep "${STOP_DELAY}"
        stop_pod
    fi
    exit "${status}"
}

# ---- Inner mode: runs inside tmux ---------------------------------------------
if [[ "${1:-}" == "__run" ]]; then
    # shellcheck disable=SC1090
    source "$2"
    run_training
fi

# ---- Launcher mode: pre-flight checks, then start tmux -------------------------
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/runs}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
CLIP_LEN="${CLIP_LEN:-4}"
SAMPLING_RATE="${SAMPLING_RATE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"          # per GPU
NUM_EPOCHS="${NUM_EPOCHS:-300}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEMPERATURE="${TEMPERATURE:-0.05}"
LR="${LR:-0.1}"
AUTO_STOP="${AUTO_STOP:-1}"
STOP_DELAY="${STOP_DELAY:-300}"         # seconds between training end and pod stop
SESSION="${SESSION:-ssl-pretrain}"
# Timestamp in the name: main.py deletes an existing run folder with the same name.
MODEL_NAME="${MODEL_NAME:-echonet_lvh_clip${CLIP_LEN}_sr${SAMPLING_RATE}_$(date +%Y%m%d-%H%M%S)}"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}.log"

command -v tmux >/dev/null 2>&1 \
    || die "tmux not found; install it with: apt-get update && apt-get install -y tmux"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    die "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION"
fi

# The container disk is wiped when the pod stops, so outputs must live on /workspace.
if [[ "${ALLOW_NON_WORKSPACE:-0}" != "1" ]]; then
    for path in "$REPO_DIR" "$OUTPUT_DIR" "$LOG_DIR"; do
        [[ "$path" == /workspace/* ]] \
            || die "$path is not under /workspace and would be lost when the pod stops (override: ALLOW_NON_WORKSPACE=1)"
    done
fi

[[ -f "${DATA_DIR}/MeasurementsList.csv" ]] || die "${DATA_DIR}/MeasurementsList.csv not found"
[[ ! -e "${OUTPUT_DIR}/${MODEL_NAME}" ]] \
    || die "${OUTPUT_DIR}/${MODEL_NAME} exists and main.py would delete it; set another MODEL_NAME"
grep -q -- "--dataset" "${REPO_DIR}/ssl_pretraining/main.py" \
    || die "ssl_pretraining/main.py has no --dataset option yet"
[[ -x "${REPO_DIR}/.venv/bin/python" ]] || die "no .venv found; run: uv sync --frozen"

N_GPU="${N_GPU:-$("${REPO_DIR}/.venv/bin/python" -c 'import torch; print(torch.cuda.device_count())')}"
[[ "$N_GPU" -ge 1 ]] || die "PyTorch sees no GPUs"

if [[ "$AUTO_STOP" == "1" ]]; then
    load_runpod_env
    [[ -n "${RUNPOD_POD_ID:-}" ]] || die "RUNPOD_POD_ID is not set; cannot auto-stop (override: AUTO_STOP=0)"
    command -v runpodctl >/dev/null 2>&1 || [[ -n "${RUNPOD_API_KEY:-}" ]] \
        || die "neither runpodctl nor RUNPOD_API_KEY is available; cannot auto-stop (override: AUTO_STOP=0)"
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Save resolved settings for the tmux session (and as a record of the run).
# The API key is not written; the inner session reloads it if needed.
ENV_FILE="${LOG_DIR}/${MODEL_NAME}.env"
RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
declare -p REPO_DIR DATA_DIR OUTPUT_DIR LOG_DIR LOG_FILE CLIP_LEN SAMPLING_RATE BATCH_SIZE \
    NUM_EPOCHS SAVE_FREQ TEMPERATURE LR AUTO_STOP STOP_DELAY MODEL_NAME N_GPU RUNPOD_POD_ID \
    >"$ENV_FILE"
chmod 600 "$ENV_FILE"

tmux new-session -d -s "$SESSION" "bash $(printf '%q' "$SCRIPT_PATH") __run $(printf '%q' "$ENV_FILE")"

cat <<EOF
Started '${MODEL_NAME}' in tmux session '${SESSION}' (${N_GPU} GPU(s), batch ${BATCH_SIZE}/GPU).
  Attach:   tmux attach -t ${SESSION}     (detach: Ctrl-b d)
  Log:      tail -f ${LOG_FILE}
  Outputs:  ${OUTPUT_DIR}/${MODEL_NAME}
  Auto-stop: $([[ "$AUTO_STOP" == "1" ]] && echo "on, ${STOP_DELAY}s after training ends" || echo "off")
You can now disconnect and close your computer.
EOF
