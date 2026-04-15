#!/bin/bash

# 基础参数
# 用法：
#   DATASET=ALOI bash ALOI.sh
#   bash ALOI.sh ALOI
# 默认：ALOI
DATASET="Caltech-5V"
BASE_DIR="./results/${DATASET}/beta_tau_search"

# -------------------- 日志落盘（避免屏幕回显导致远程 IDE 卡顿） --------------------
# 需求：
#   - 实验日志：   ./logs/alpha_beta_tau_search/${DATASET}/${DATASET}_*.log
#   - 管理日志：   ./logs/alpha_beta_tau_search/${DATASET}/manager.log
#   - 不使用时间戳子目录；同参实验可在同一个日志文件里继续追加（不覆盖）
#
# 可通过环境变量覆盖 LOG_DIR（例如 LOG_DIR=/data/logs/alpha_beta_tau_search/${DATASET}）
LOG_DIR="${LOG_DIR:-./logs/beta_tau_search/${DATASET}}"
MANAGER_LOG="${LOG_DIR}/manager.log"

# 新超参数设置
# 困难正样本参数
HP_MAXITER="5"
HP_ALPHA="0.05"

# 困难负样本参数（这里保留BETA_VALUES作为网格搜索参数）
HN_MAXITER="5"
HN_BETA="0.05"

# ==================== 修改点1：定义完整的超参搜索空间 ====================
# 对比损失温度系数搜索（逆序）
HARD_TAU_VALUES=("0" "10000" "2000" "1000" "200" "100" "10" "1")

# beta网格搜索值（作为对比损失中的权重或其他用途）（逆序）
BETA_VALUES=("0" "0.0001" "0.0005" "0.001" "0.005" "0.01" "0.1" "1")

# ==================== 修改点2：定义已跑实验的组合 ====================
# 从图片中提取的已跑实验
declare -A COMPLETED_EXPERIMENTS
# 格式：COMPLETED_EXPERIMENTS["beta,tau"]=1
# 使用关联数组存储已跑实验

# 手动输入已跑实验（根据图片）
COMPLETED_EXPERIMENTS["0,0"]=1
COMPLETED_EXPERIMENTS["0.1,10000"]=1
COMPLETED_EXPERIMENTS["0.0001,0"]=1
COMPLETED_EXPERIMENTS["1,10000"]=1
COMPLETED_EXPERIMENTS["1,0"]=1
COMPLETED_EXPERIMENTS["0.01,0"]=1
COMPLETED_EXPERIMENTS["0.0001,10000"]=1
COMPLETED_EXPERIMENTS["0.005,10000"]=1
COMPLETED_EXPERIMENTS["0.001,0"]=1
COMPLETED_EXPERIMENTS["0,10000"]=1
COMPLETED_EXPERIMENTS["0.005,0"]=1
COMPLETED_EXPERIMENTS["0.0005,10000"]=1
COMPLETED_EXPERIMENTS["0.0005,0"]=1
COMPLETED_EXPERIMENTS["0.1,0"]=1
COMPLETED_EXPERIMENTS["0.01,10000"]=1
COMPLETED_EXPERIMENTS["0.001,10000"]=1

# ==================== 修改点3：自动生成需要补跑的组合 ====================
declare -a PARAM_COMBINATIONS=()

echo "[INFO] 完整搜索空间: ${#BETA_VALUES[@]}个beta值 × ${#HARD_TAU_VALUES[@]}个tau值 = $(( ${#BETA_VALUES[@]} * ${#HARD_TAU_VALUES[@]} ))个组合"
echo "[INFO] 已跑实验数: ${#COMPLETED_EXPERIMENTS[@]}"

# 遍历所有可能的组合，找出未跑的实验
for BETA in "${BETA_VALUES[@]}"; do
    for TAU in "${HARD_TAU_VALUES[@]}"; do
        # 检查这个组合是否已跑
        key="${BETA},${TAU}"
        if [[ -z "${COMPLETED_EXPERIMENTS[${key}]}" ]]; then
            # 未跑，加入补跑列表
            PARAM_COMBINATIONS+=("${BETA} ${TAU}")
            echo "[INFO] 需要补跑: beta=${BETA}, tau=${TAU}"
        fi
    done
done

echo "[INFO] 需要补跑的实验数: ${#PARAM_COMBINATIONS[@]}"

# 如果不需要补跑，直接退出
if [ ${#PARAM_COMBINATIONS[@]} -eq 0 ]; then
    echo "[INFO] 所有实验都已跑完，无需补跑。"
    exit 0
fi

# 新增 ALPHA 搜索
ALPHA_VALUES=("1")

# 实验名称（用于邮件通知）
EXPERIMENT_NAME="${DATASET}_MissingExperimentsCompletion"

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
echo "[INFO] 本次将运行以下 ${#PARAM_COMBINATIONS[@]} 个需要补的实验："
for combo in "${PARAM_COMBINATIONS[@]}"; do
    BETA_WEIGHT=$(echo "$combo" | awk '{print $1}')
    HARD_TAU=$(echo "$combo" | awk '{print $2}')
    echo "[INFO]   beta=${BETA_WEIGHT}, tau=${HARD_TAU}"
done

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

    # 检查日志文件是否已存在
    if [ -f "${LOG_FILE}" ]; then
        echo "[WARNING] 日志文件已存在: ${LOG_FILE}"
        echo "[WARNING] 同参实验将在该日志文件中追加输出"
    fi

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

        # 获取所有 GPU 的显存信息
        gpu_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)

        # 创建一个数组来存储所有可用的 GPU（按优先级排序）
        declare -a available_gpus=()

        # 首先检查 GPU 1（最高优先级）
        gpu1_info=$(echo "$gpu_info" | grep "^1," | head -1)
        if [ -n "$gpu1_info" ]; then
            gid=$(echo "$gpu1_info" | cut -d',' -f1 | tr -d ' \t\r')
            free=$(echo "$gpu1_info" | cut -d',' -f2 | tr -d ' \t\r')

            # 先清理过期的冷却锁
            gpu_lock_cleanup_if_expired "${gid}"

            # 检查显存是否足够
            if [ $((free+0)) -ge "${MIN_FREE_MEM_MB}" ]; then
                # 检查冷却锁状态
                lockf="$(gpu_lock_path "${gid}")"
                if [ ! -f "${lockf}" ]; then
                    # GPU 1 可用且不在冷却期，直接选择
                    if gpu_lock_try_acquire "${gid}"; then
                        echo "[INFO] 选择 GPU 1 (显存: ${free}MiB, 可用)" >&2
                        echo "${gid}"
                        return 0
                    fi
                else
                    # GPU 1 在冷却期，记录状态
                    echo "[INFO] GPU 1 在冷却期内，必须等待" >&2
                    # 等待冷却期结束
                    sleep "${GPU_COOLDOWN_SEC}"
                    continue  # 重新开始循环，再次检查 GPU 1
                fi
            else
                echo "[INFO] GPU 1 显存不足 (可用: ${free}MiB, 需要: ${MIN_FREE_MEM_MB}MiB)" >&2
            fi
        else
            echo "[WARNING] GPU 1 不存在或无法访问" >&2
        fi

        # 只有当 GPU 1 确实不可用（显存不足或不存在）时，才考虑其他 GPU
        # 注意：如果 GPU 1 在冷却期，我们已经在上面的逻辑中等待了，不会执行到这里

        # 检查其他 GPU（按索引顺序）
        echo "[INFO] GPU 1 不可用，检查其他 GPU..." >&2
        for gid in 0 2 3 4 5 6 7; do  # 假设最多8块卡，排除1
            # 获取该 GPU 的信息
            gpu_info_line=$(echo "$gpu_info" | grep "^${gid}," | head -1)
            if [ -z "$gpu_info_line" ]; then
                continue  # GPU 不存在
            fi

            free=$(echo "$gpu_info_line" | cut -d',' -f2 | tr -d ' \t\r')

            # 清理过期的冷却锁
            gpu_lock_cleanup_if_expired "${gid}"

            # 检查显存是否足够
            if [ $((free+0)) -ge "${MIN_FREE_MEM_MB}" ]; then
                # 检查冷却锁状态
                lockf="$(gpu_lock_path "${gid}")"
                if [ ! -f "${lockf}" ]; then
                    # GPU 可用且不在冷却期
                    available_gpus+=("${gid}:${free}")
                else
                    echo "[INFO] GPU ${gid} 在冷却期内" >&2
                fi
            else
                echo "[INFO] GPU ${gid} 显存不足 (可用: ${free}MiB)" >&2
            fi
        done

        # 如果有可用的其他 GPU
        if [ ${#available_gpus[@]} -gt 0 ]; then
            # 选择第一个可用的 GPU（按优先级排序）
            first_gpu="${available_gpus[0]}"
            gid=$(echo "$first_gpu" | cut -d':' -f1)
            free=$(echo "$first_gpu" | cut -d':' -f2)

            if gpu_lock_try_acquire "${gid}"; then
                echo "[INFO] GPU 1 不可用，选择 GPU ${gid} (显存: ${free}MiB)" >&2
                echo "${gid}"
                return 0
            fi
        fi

        # 没有可用的 GPU，等待后重试
        echo "No GPU available (free>=${MIN_FREE_MEM_MB}MiB). Waiting ${GPU_CHECK_INTERVAL}s..." >&2
        sleep "${GPU_CHECK_INTERVAL}"
        # 继续循环，重新从 GPU 1 开始检查
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

# ==================== 修改点4：遍历需要补的参数组合 ====================
echo "[INFO] 开始运行需要补的 ${#PARAM_COMBINATIONS[@]} 个实验..."
for combo in "${PARAM_COMBINATIONS[@]}"; do
    # 解析参数组合
    BETA_WEIGHT=$(echo "$combo" | awk '{print $1}')
    HARD_TAU=$(echo "$combo" | awk '{print $2}')

    # ALPHA固定为1
    ALPHA="1"

    # 先等并行名额
    throttle

    # 再选 GPU（阻塞直到找到满足阈值的卡；没有 nvidia-smi 则返回空）
    gpu_id="$(pick_gpu_blocking)"
    # Defensive: strip whitespace/newlines and make sure it is a numeric GPU id.
    gpu_id="$(echo "${gpu_id}" | tr -d ' \t\r\n')"
    if [[ -n "${gpu_id}" && ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] pick_gpu_blocking returned invalid gpu_id='${gpu_id}'. Retrying..." >&2
        # Avoid accidentally running on CPU due to a corrupted CUDA_VISIBLE_DEVICES.
        sleep "${GPU_CHECK_INTERVAL}"
        continue
    fi

    echo "[INFO] 启动实验: beta=${BETA_WEIGHT}, tau=${HARD_TAU}, alpha=${ALPHA}"
    # 启动任务
    launch_experiment "${gpu_id}" "${HARD_TAU}" "${ALPHA}" "${BETA_WEIGHT}"
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

# 输出补跑统计
echo ""
echo "==================== 补跑统计 ===================="
echo "总实验组合: $(( ${#BETA_VALUES[@]} * ${#HARD_TAU_VALUES[@]} ))"
echo "已跑实验: ${#COMPLETED_EXPERIMENTS[@]}"
echo "本次补跑: ${#PARAM_COMBINATIONS[@]}"
echo "补跑完成时间: ${GLOBAL_END_TIME}"
echo "总运行时长: ${DURATION}"
echo "日志目录: ${LOG_DIR}"
echo "=================================================="

echo "所有需要补的实验已完成!"
echo "Log files are saved in: ${LOG_DIR}"