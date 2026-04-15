#!/bin/bash

# 基础参数
# 用法：
#   DATASET=ALOI bash ALOI.sh
#   bash ALOI.sh ALOI
# 默认：ALOI
DATASET="COIL20"
BASE_DIR="./results/${DATASET}/beta_tau_search_new"

# -------------------- 日志落盘（避免屏幕回显导致远程 IDE 卡顿） --------------------
# 需求：
#   - 实验日志：   ./logs/alpha_beta_tau_search/${DATASET}/${DATASET}_*.log
#   - 管理日志：   ./logs/alpha_beta_tau_search/${DATASET}/manager.log
#   - 不使用时间戳子目录；同参实验可在同一个日志文件里继续追加（不覆盖）
#
# 可通过环境变量覆盖 LOG_DIR（例如 LOG_DIR=/data/logs/alpha_beta_tau_search/${DATASET}）
LOG_DIR="${LOG_DIR:-./logs/beta_tau_search_new/${DATASET}}"
MANAGER_LOG="${LOG_DIR}/manager.log"

# 新超参数设置
# 困难正样本参数
HP_MAXITER="5"
HP_ALPHA="0.05"

# 困难负样本参数（这里保留BETA_VALUES作为网格搜索参数）
HN_MAXITER="5"
HN_BETA="0.05"

# 对比损失温度系数搜索
# HARD_TAU_VALUES=("1" "10" "100" "200" "1000" "2000" "10000" "0")

# beta网格搜索值（作为对比损失中的权重或其他用途）
# BETA_VALUES=("1" "0.1" "0.01" "0.005" "0.001" "0.0005" "0.0001" "0")

# 对比损失温度系数搜索（逆序）
HARD_TAU_VALUES=("0" "10000" "2000" "1000" "200" "100" "10" "1")

# beta网格搜索值（作为对比损失中的权重或其他用途）（逆序）
BETA_VALUES=("0" "0.0001" "0.0005" "0.001" "0.005" "0.01" "0.1" "1")

# 新增 ALPHA 搜索
ALPHA_VALUES=("1")

# 实验名称（用于邮件通知）
EXPERIMENT_NAME="${DATASET}_TauAlphaBetaGridSearch"

# 最大并行实验数
MAX_JOBS=2

# GPU 显存阈值（MiB），至少有一块 GPU 空闲显存 >= 该值时才启动新的实验
MIN_FREE_MEM_MB=4000

# GPU 检查间隔（秒）
GPU_CHECK_INTERVAL=5

# GPU 冷却锁：防止短时间内反复把任务分配到同一块卡（nvidia-smi 反映可能有延迟）
GPU_LOCK_DIR="/tmp/alpha_beta_tau_search_gpu_locks"
GPU_COOLDOWN_SEC=30
mkdir -p "${GPU_LOCK_DIR}"

# 整个脚本的全局开始时间（用于计算总运行时长）
GLOBAL_START_TIME="$(date +"%Y-%m-%d %H:%M:%S")"

# 创建日志目录
mkdir -p "${LOG_DIR}" "${BASE_DIR}"

# 关闭屏幕回显：将脚本（调度器）自身的 stdout/stderr 全部追加写入 manager.log
# 注意：单个实验的 stdout/stderr 仍会写入各自的 ${DATASET}_*.log
exec >>"${MANAGER_LOG}" 2>&1

echo "[INFO] dataset: ${DATASET}"
echo "[INFO] manager log: ${MANAGER_LOG}"
echo "[INFO] experiment logs: ${LOG_DIR}/${DATASET}_*.log"

# ---------------- GPU cooldown lock helpers ----------------
gpu_lock_path() { echo "${GPU_LOCK_DIR}/gpu_${1}.lock"; }

gpu_lock_cleanup_if_expired() {
    local gid="$1" lockf now ts age
    lockf="$(gpu_lock_path "${gid}")"
    [ -f "${lockf}" ] || return 0
    now=$(date +%s)
    ts=$(head -n 1 "${lockf}" 2>/dev/null || echo "0"); ts=${ts:-0}
    age=$((now - ts))
    [ "${age}" -ge "${GPU_COOLDOWN_SEC}" ] && rm -f "${lockf}" 2>/dev/null || true
}

gpu_lock_try_acquire() {
    local gid="$1" lockf
    lockf="$(gpu_lock_path "${gid}")"
    gpu_lock_cleanup_if_expired "${gid}"
    ( set -o noclobber; echo "$(date +%s)" > "${lockf}" ) 2>/dev/null
}

# 单次实验函数：保持原有日志和参数不变
run_single_experiment() {
    local HARD_TAU="$1"
    local ALPHA="$2"
    local BETA_WEIGHT="$3"

    echo "========================================"
    echo "Running experiment with hard_tau=${HARD_TAU}, alpha=${ALPHA}, beta_weight=${BETA_WEIGHT}"
    echo "========================================"

    # 构建参数文件名后缀（使用新参数）
    PARAM_SUFFIX="beta${BETA_WEIGHT}_tau${HARD_TAU}_alpha${ALPHA}_hpm${HP_MAXITER}_hpa${HP_ALPHA}_hnm${HN_MAXITER}_hnb${HN_BETA}"

    # 生成日志文件名（去掉小数点的下划线，避免文件名问题）
    LOG_SUFFIX=$(echo "${PARAM_SUFFIX}" | sed 's/\./_/g')
    # 需求：日志文件名以 ${DATASET}_ 开头，且同参实验可继续追加写（不覆盖）
    LOG_FILE="${LOG_DIR}/${DATASET}_${LOG_SUFFIX}.log"

    # 生成文件名（去掉小数点的下划线）
    CLEAN_SUFFIX=$(echo "${PARAM_SUFFIX}" | sed 's/\./_/g')

    # 生成模型文件路径
    PRETRAIN_WEIGHTS="${BASE_DIR}/ae_weights_${CLEAN_SUFFIX}.pth"
    WEIGHTS="${BASE_DIR}/model_final_${CLEAN_SUFFIX}.pth"

    # 创建结果目录（如果不存在）
    mkdir -p "${BASE_DIR}"

    echo "Experiment parameters:"
    echo "  Dataset: ${DATASET}"
    echo "  Beta weight: ${BETA_WEIGHT}"
    echo "  Hard tau (τ): ${HARD_TAU}"
    echo "  Alpha: ${ALPHA}"
    echo "  Hard positive maxiter: ${HP_MAXITER}"
    echo "  Hard positive alpha: ${HP_ALPHA}"
    echo "  Hard negative maxiter: ${HN_MAXITER}"
    echo "  Hard negative beta: ${HN_BETA}"
    echo "  Pretrain weights: ${PRETRAIN_WEIGHTS}"
    echo "  Final weights: ${WEIGHTS}"
    echo "  Log file: ${LOG_FILE}"
    echo ""

    # 开始时间
    START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Start time: ${START_TIME}"

    # 将实验输出仅写入该实验的日志文件（不回显到屏幕/manager.log，避免刷屏）
    echo "=================== Experiment Start ===================" >> "${LOG_FILE}"
    echo "Start time: ${START_TIME}" >> "${LOG_FILE}"
    echo "Parameters:" >> "${LOG_FILE}"
    echo "  Dataset: ${DATASET}" >> "${LOG_FILE}"
    echo "  Beta weight: ${BETA_WEIGHT}" >> "${LOG_FILE}"
    echo "  Hard tau (τ): ${HARD_TAU}" >> "${LOG_FILE}"
    echo "  Alpha: ${ALPHA}" >> "${LOG_FILE}"
    echo "  Hard positive maxiter: ${HP_MAXITER}" >> "${LOG_FILE}"
    echo "  Hard positive alpha: ${HP_ALPHA}" >> "${LOG_FILE}"
    echo "  Hard negative maxiter: ${HN_MAXITER}" >> "${LOG_FILE}"
    echo "  Hard negative beta: ${HN_BETA}" >> "${LOG_FILE}"
    echo "======================================================" >> "${LOG_FILE}"

    # Optional GPU preflight: if CUDA_VISIBLE_DEVICES is set, ensure the process can really see a GPU.
    # This prevents silent CPU fallback when CUDA_VISIBLE_DEVICES is corrupted or GPU is unavailable.
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        python - <<'PY' >> "${LOG_FILE}" 2>&1
import os, sys
try:
    import torch
    print("[GPU-PREFLIGHT] CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[GPU-PREFLIGHT] torch=", torch.__version__)
    print("[GPU-PREFLIGHT] torch.version.cuda=", getattr(torch.version, "cuda", None))
    print("[GPU-PREFLIGHT] cuda.is_available=", torch.cuda.is_available())
    print("[GPU-PREFLIGHT] device_count=", torch.cuda.device_count())
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        sys.exit(99)
    print("[GPU-PREFLIGHT] device0=", torch.cuda.get_device_name(0))
except Exception as e:
    print("[GPU-PREFLIGHT] exception:", repr(e))
    sys.exit(99)
PY

        preflight_ec=$?
        if [[ ${preflight_ec} -ne 0 ]]; then
            echo "[ERROR] GPU preflight failed (exit=${preflight_ec}). Refusing to run on CPU." >> "${LOG_FILE}"
            EXIT_CODE=${preflight_ec}
            END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
            echo "======================================================" >> "${LOG_FILE}"
            echo "End time: ${END_TIME}" >> "${LOG_FILE}"
            echo "Exit code: ${EXIT_CODE}" >> "${LOG_FILE}"
            echo "==================== Experiment End ===================" >> "${LOG_FILE}"
            return "${preflight_ec}"
        fi
    fi

    # 执行命令，将 stdout/stderr 追加写入日志文件
    PYTHONUNBUFFERED=1 python -u main.py \
        -d "${DATASET}" \
        --beta "${BETA_WEIGHT}" \
        --alpha "${ALPHA}" \
        --pretrain-weights "${PRETRAIN_WEIGHTS}" \
        --weights "${WEIGHTS}" \
        --hard-tau "${HARD_TAU}" \
        --hp-maxiter "${HP_MAXITER}" \
        --hp-alpha "${HP_ALPHA}" \
        --hn-maxiter "${HN_MAXITER}" \
        --hn-beta "${HN_BETA}" \
        >> "${LOG_FILE}" 2>&1

    # 获取命令退出状态
    EXIT_CODE=$?

    # 结束时间
    END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    echo "End time: ${END_TIME}"

    # 将结束信息写入日志
    echo "======================================================" >> "${LOG_FILE}"
    echo "End time: ${END_TIME}" >> "${LOG_FILE}"
    echo "Exit code: ${EXIT_CODE}" >> "${LOG_FILE}"
    echo "==================== Experiment End ===================" >> "${LOG_FILE}"

    # 检查上一个命令是否成功
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "Experiment with hard_tau=${HARD_TAU}, alpha=${ALPHA}, beta_weight=${BETA_WEIGHT} completed successfully!"
        echo "Log saved to: ${LOG_FILE}"
    else
        echo "Experiment with hard_tau=${HARD_TAU}, alpha=${ALPHA}, beta_weight=${BETA_WEIGHT} failed with exit code: ${EXIT_CODE}"
        echo "Check log file for details: ${LOG_FILE}"
        # 可以选择是否继续执行后续实验
        # exit 1
    fi

    echo ""
}

# ---------------- 稳定并行调度：PID 数组 + wait -n ----------------
PIDS=()

cleanup_all() {
    # 终止所有仍在跑的后台实验（及其子进程）
    for pid in "${PIDS[@]}"; do
        kill -0 "${pid}" 2>/dev/null || continue
        kill "${pid}" 2>/dev/null
    done

    # 给一点时间退出，不退出则 KILL
    sleep 1
    for pid in "${PIDS[@]}"; do
        kill -0 "${pid}" 2>/dev/null || continue
        kill -9 "${pid}" 2>/dev/null || true
    done

    # 释放 GPU 冷却锁（避免下次启动被旧锁影响）
    rm -f "${GPU_LOCK_DIR}"/gpu_*.lock 2>/dev/null || true
}

# Ctrl+C / kill 时清理
trap 'echo ""; echo "Caught interrupt, terminating all running experiments..."; cleanup_all; exit 130' INT TERM

# 清理已结束的 PID（维护 PIDS 数组）
refresh_pids() {
    local new=()
    for p in "${PIDS[@]}"; do
        if kill -0 "${p}" 2>/dev/null; then
            new+=("${p}")
        fi
    done
    PIDS=("${new[@]}")
}

# 达到并行上限就等任意一个结束
throttle() {
    while true; do
        refresh_pids
        if [ "${#PIDS[@]}" -lt "${MAX_JOBS}" ]; then
            return 0
        fi
        # 等一个结束（bash 4.3+）
        wait -n
    done
}

# 选一块显存够用的 GPU（返回 GPU index；若找不到则阻塞等待；若没有 nvidia-smi 则返回空字符串）
pick_gpu_blocking() {
    while true; do
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo ""
            return 0
        fi

        # 选第一块满足：free >= 阈值 且 能获得 cooldown 锁 的 GPU
        # 这能避免短时间连续启动时被分配到同一块卡（显存/进程状态还没来得及更新）
        while IFS=',' read -r gid free; do
            gid="$(echo "${gid}" | tr -d ' \t\r')"
            free="$(echo "${free}" | tr -d ' \t\r')"
            [ -z "${gid}" ] && continue
            [ $((free+0)) -lt "${MIN_FREE_MEM_MB}" ] && continue

            if gpu_lock_try_acquire "${gid}"; then
                echo "${gid}"
                return 0
            fi
        done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)

        # IMPORTANT: this function is called via command-substitution, so stdout MUST contain ONLY the GPU id.
        # Send status messages to stderr (they will still be captured in manager.log via exec redirection).
        echo "No GPU available (free>=${MIN_FREE_MEM_MB}MiB + cooldown ${GPU_COOLDOWN_SEC}s). Waiting ${GPU_CHECK_INTERVAL}s..." >&2
        sleep "${GPU_CHECK_INTERVAL}"
    done
}

# 启动一个实验，记录 PID
launch_experiment() {
    local gpu_id="$1"
    local HARD_TAU="$2"
    local ALPHA="$3"
    local BETA_WEIGHT="$4"

    if [ -n "${gpu_id}" ]; then
        echo "Launching experiment on GPU ${gpu_id} (hard_tau=${HARD_TAU}, alpha=${ALPHA}, beta_weight=${BETA_WEIGHT})"
        CUDA_VISIBLE_DEVICES="${gpu_id}" run_single_experiment "${HARD_TAU}" "${ALPHA}" "${BETA_WEIGHT}" &
    else
        echo "Launching experiment without explicit GPU binding (hard_tau=${HARD_TAU}, alpha=${ALPHA}, beta_weight=${BETA_WEIGHT})"
        run_single_experiment "${HARD_TAU}" "${ALPHA}" "${BETA_WEIGHT}" &
    fi

    PIDS+=($!)
}

# ---------------- 遍历所有组合：HARD_TAU × ALPHA × BETA_WEIGHT ----------------
for HARD_TAU in "${HARD_TAU_VALUES[@]}"
do
    for ALPHA in "${ALPHA_VALUES[@]}"
    do
        for BETA_WEIGHT in "${BETA_VALUES[@]}"
        do
            # 先等并行名额
            throttle

            # 再选 GPU（阻塞直到找到满足阈值的卡；没有 nvidia-smi 则返回空）
            # NOTE: pick_gpu_blocking prints only the GPU id to stdout; all status messages go to stderr/manager.log.
            gpu_id="$(pick_gpu_blocking)"
            # Defensive: strip whitespace/newlines and make sure it is a numeric GPU id.
            gpu_id="$(echo "${gpu_id}" | tr -d ' \t\r\n')"
            if [[ -n "${gpu_id}" && ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
                echo "[ERROR] pick_gpu_blocking returned invalid gpu_id='${gpu_id}'. Retrying..." >&2
                # Avoid accidentally running on CPU due to a corrupted CUDA_VISIBLE_DEVICES.
                sleep "${GPU_CHECK_INTERVAL}"
                continue
            fi

            # 启动任务
            launch_experiment "${gpu_id}" "${HARD_TAU}" "${ALPHA}" "${BETA_WEIGHT}"
        done
    done
done

# 等待所有后台任务结束
while true; do
    refresh_pids
    [ "${#PIDS[@]}" -eq 0 ] && break
    wait -n
done

# 发送完成邮件
echo ""
echo "Sending experiment completion email..."

# 全局结束时间取发送邮件时刻
GLOBAL_END_TIME="$(date +"%Y-%m-%d %H:%M:%S")"

# 计算总运行时长（秒）
START_TS=$(date -d "${GLOBAL_START_TIME}" +%s)
END_TS=$(date -d "${GLOBAL_END_TIME}" +%s)
DURATION_SEC=$((END_TS - START_TS))

# 转成 HH:MM:SS 格式
HOURS=$((DURATION_SEC / 3600))
MINUTES=$(((DURATION_SEC % 3600) / 60))
SECONDS=$((DURATION_SEC % 60))
DURATION=$(printf "%02d:%02d:%02d" "$HOURS" "$MINUTES" "$SECONDS")

python ~/send_experiment_email.py \
    --experiment-name "${EXPERIMENT_NAME}" \
    --start-time "${GLOBAL_START_TIME}" \
    --end-time "${GLOBAL_END_TIME}" \
    --duration "${DURATION}"

MAIL_EXIT_CODE=$?

if [ ${MAIL_EXIT_CODE} -eq 0 ]; then
    echo "Email sent successfully!"
else
    echo "Failed to send email. Check configuration."
fi

echo "All experiments completed!"
echo "Log files are saved in: ${LOG_DIR}"
