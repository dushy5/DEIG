#!/bin/bash

# ============================================
# 多卡并行推理脚本 - MIG Bench 评估
# ============================================

# 默认配置参数
NUM_GPUS=${NUM_GPUS:-8}                    # GPU数量，默认8卡
OUTPUT_FOLDER=${OUTPUT_FOLDER:-"generation_samples/migbench"}  # 输出目录
CONFIG=${CONFIG:-"configs/config.yaml"}    # 配置文件路径
SEED=${SEED:-123}                           # 随机种子
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7.5}      # guidance scale
# 评估相关参数
RUN_EVAL=${RUN_EVAL:-true}                 # 是否运行评估
NEED_CLIP_SCORE=${NEED_CLIP_SCORE:-true}
NEED_SUCCESS_RATIO=${NEED_SUCCESS_RATIO:-true}
NEED_LOCAL_CLIP=${NEED_LOCAL_CLIP:-false}
NEED_MIOU_SCORE=${NEED_MIOU_SCORE:-true}
NEED_INSTANCE_SUCCESS_RATIO=${NEED_INSTANCE_SUCCESS_RATIO:-true}
MIOU_THRESHOLD=${MIOU_THRESHOLD:-0.5}
METRIC_NAME=${METRIC_NAME:-"migbench"}

# 是否使用FP16
USE_FP16=${USE_FP16:-false}

# 打印配置
echo "============================================"
echo "多卡并行推理配置 - MIG Bench"
echo "============================================"
echo "GPU数量: ${NUM_GPUS}"
echo "输出目录: ${OUTPUT_FOLDER}"
echo "配置文件: ${CONFIG}"
echo "随机种子: ${SEED}"
echo "使用FP16: ${USE_FP16}"
echo "运行评估: ${RUN_EVAL}"
echo "============================================"

# 创建输出目录
mkdir -p ${OUTPUT_FOLDER}

# 创建日志目录
LOG_DIR="${OUTPUT_FOLDER}/logs"
mkdir -p ${LOG_DIR}

# 构建基础命令参数
BASE_ARGS="--folder ${OUTPUT_FOLDER} --config ${CONFIG} --seed ${SEED} --guidance_scale ${GUIDANCE_SCALE} --batch_size 1 --num_jobs ${NUM_GPUS}"

if [ "${USE_FP16}" = true ]; then
    BASE_ARGS="${BASE_ARGS} --fp16"
fi

if [ -n "${NEGATIVE_PROMPT}" ]; then
    BASE_ARGS="${BASE_ARGS} --negative_prompt \"${NEGATIVE_PROMPT}\""
fi

# 存储所有后台进程的PID
declare -a PIDS

# 启动多卡并行推理
echo ""
echo "开始启动 ${NUM_GPUS} 张GPU并行推理..."
echo ""

for ((i=0; i<${NUM_GPUS}; i++)); do
    echo "启动 GPU ${i} 的推理任务 (job_index=${i})..."
    
    # 设置CUDA可见设备并启动后台任务
    CUDA_VISIBLE_DEVICES=${i} python eval/eval_migbench.py \
        ${BASE_ARGS} \
        --job_index ${i} \
        > "${LOG_DIR}/gpu_${i}.log" 2>&1 &
    
    PIDS[$i]=$!
    echo "  GPU ${i} 进程PID: ${PIDS[$i]}"
done

echo ""
echo "所有推理任务已启动，等待完成..."
echo ""

# 等待所有进程完成
FAILED=0
for ((i=0; i<${NUM_GPUS}; i++)); do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "警告: GPU ${i} 的任务失败，退出码: ${EXIT_CODE}"
        echo "查看日志: ${LOG_DIR}/gpu_${i}.log"
        FAILED=$((FAILED + 1))
    else
        echo "GPU ${i} 的推理任务完成"
    fi
done

echo ""
if [ ${FAILED} -gt 0 ]; then
    echo "警告: ${FAILED} 个任务失败"
else
    echo "所有推理任务完成!"
fi
echo ""

# 运行评估（仅在第0卡上运行）
if [ "${RUN_EVAL}" = true ]; then
    echo "============================================"
    echo "开始运行评估..."
    echo "============================================"
    
    # 构建评估参数
    EVAL_ARGS="--folder ${OUTPUT_FOLDER} --config ${CONFIG} --seed ${SEED} --batch_size 1 --num_jobs 1 --job_index 0 --run_eval"
    
    if [ "${USE_FP16}" = true ]; then
        EVAL_ARGS="${EVAL_ARGS} --fp16"
    fi
    
    if [ "${NEED_CLIP_SCORE}" = true ]; then
        EVAL_ARGS="${EVAL_ARGS} --need_clip_score"
    fi
    
    if [ "${NEED_SUCCESS_RATIO}" = true ]; then
        EVAL_ARGS="${EVAL_ARGS} --need_sucess_ratio"
    fi
    
    if [ "${NEED_LOCAL_CLIP}" = true ]; then
        EVAL_ARGS="${EVAL_ARGS} --need_local_clip"
    fi
    
    if [ "${NEED_MIOU_SCORE}" = true ]; then
        EVAL_ARGS="${EVAL_ARGS} --need_miou_score"
    fi
    
    if [ "${NEED_INSTANCE_SUCCESS_RATIO}" = true ]; then
        EVAL_ARGS="${EVAL_ARGS} --need_instance_sucess_ratio"
    fi
    
    EVAL_ARGS="${EVAL_ARGS} --miou_threshold ${MIOU_THRESHOLD} --metric_name ${METRIC_NAME}"
    
    echo "评估命令: python eval/eval_migbench.py ${EVAL_ARGS}"
    echo ""
    
    # 在GPU 0上运行评估
    CUDA_VISIBLE_DEVICES=0 python eval/eval_migbench.py ${EVAL_ARGS}
    
    echo ""
    echo "评估完成!"
    echo "结果保存在: ${OUTPUT_FOLDER}/metric_${METRIC_NAME}.json"
fi

echo ""
echo "============================================"
echo "全部任务完成!"
echo "============================================"
