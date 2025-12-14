#!/bin/bash

# ==============================================================================
# 多进程多GPU并行处理脚本 (后台作业控制版本)
# - 修复了xargs导致GPU任务分配不均的问题
# - 为每个GPU创建独立的任务队列
#
# 修改:
# - 将配置文件名从硬编码改为从命令行参数读取
# ==============================================================================

# --- 1. 用户配置区 ---
# [MODIFIED] 检查参数数量，现在需要2个参数
if [ $# -ne 2 ]; then
    echo "错误: 请提供父目录路径和配置文件名作为参数。"
    echo "用法: $0 /path/to/parent_directory config_file.ini"
    exit 1
fi

PARENT_DIR="$1"
CONFIG_FILE="$2" # [NEW] 从第二个参数读取配置文件名

# 检查父目录是否存在
if [ ! -d "$PARENT_DIR" ]; then
    echo "错误: 父目录不存在: $PARENT_DIR"
    exit 1
fi

# [NEW] 检查配置文件是否存在
if [ ! -f "config/$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi


# --- 2. GPU和进程数配置 ---
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -eq 0 ]; then
    echo "错误: 未检测到任何NVIDIA GPU，或者 nvidia-smi 命令执行失败。"
    exit 1
fi
echo "[INFO] 检测到 ${NUM_GPUS} 块GPU。"

# --- 统计功能设置 ---
STATUS_DIR="/tmp/batch_arkitscene_status_$$" # 使用 $$ 保证临时目录唯一
mkdir -p "$STATUS_DIR"
# trap命令确保脚本在退出时（无论是正常结束还是被中断）都能执行清理操作
trap 'echo "正在清理临时文件..."; rm -rf "$STATUS_DIR"; echo "清理完毕。"' EXIT

# --- 3. 核心处理函数 (仅接收工作区路径) ---
# GPU ID 和 CONFIG_FILE 将通过环境变量传入
process_single_workspace() {
    local workspace="$1"
    # 函数内部不再接收gpu_id，直接从环境变量读取
    local gpu_id=$CUDA_VISIBLE_DEVICES
    local ws_start_time=$SECONDS

    local cad_retrieval_dir="$workspace/intermediate/HOC_Search/CAD_retrieval"
    local success_flag_file="${cad_retrieval_dir}/cad_object_label.txt"
    # voxel 路径
    local voxels_dir="${cad_retrieval_dir}/preprocessed_voxels_with_cad"

    # 初始化跳过标志 (0=不跳过/需重跑, 1=跳过/成功)
    local should_skip=0
    local skip_reason=""

    # 检查成功标志文件 'cad_object_label.txt' 是否存在
    if [ -f "$success_flag_file" ]; then
        # 如果文件存在，说明已经成功处理过，直接跳过
        echo "[GPU ${gpu_id}] 检测到成功标志 '${success_flag_file##*/}'，跳过: $workspace"
        touch "$STATUS_DIR/$(basename "$workspace").success"
        return 0
    else
        # 如果文件不存在，说明处理未完成或失败。
        # 在重新处理之前，先删除旧的CAD_retrieval文件夹以清理环境。
        if [ -d "$cad_retrieval_dir" ]; then
            echo "[GPU ${gpu_id}] 未检测到成功标志。正在清理旧的目录: $cad_retrieval_dir"
            rm -rf "$cad_retrieval_dir"
        fi
    fi

#    echo "[GPU ${gpu_id}] 开始处理工作区: $workspace"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [GPU ${gpu_id}] 开始处理工作区: $workspace"

    local log_dir="$workspace/intermediate/HOC_Search/CAD_retrieval/logs"
    mkdir -p "$log_dir"
    local log_file="${log_dir}/cad_retrieval_$(date +%Y%m%d_%H%M%S).log"

    local scene_id=$(basename "$workspace")

    # [MODIFIED] 使用从参数传入的 $CONFIG_FILE 变量
    python HOC_search/CAD_retrieval_HOC_search_ArkitScene.py \
        --config "$CONFIG_FILE" \
        --scene_id "$scene_id" > "$log_file" 2>&1

    local exit_code=$?
    local ws_duration=$((SECONDS - ws_start_time))
    if [ $exit_code -eq 0 ]; then
#        echo "[GPU ${gpu_id}] [SUCCESS] 成功处理: $workspace"
        echo "$(date '+%Y-%m-%d %H:%M:%S') [GPU ${gpu_id}] [SUCCESS] 成功处理: $workspace"
        touch "$STATUS_DIR/$(basename "$workspace").success"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [GPU ${gpu_id}] [ERROR] 处理失败: $workspace"
#        echo "[GPU ${gpu_id}] [ERROR] 处理失败: $workspace"
    fi
    printf "[GPU ${gpu_id}] --- 工作区 '%s' 处理耗时: %d 分 %d 秒 ---\n" "$(basename "$workspace")" $((ws_duration / 60)) $((ws_duration % 60))
}

# [MODIFIED] 导出函数和需要的环境变量，包括新的 CONFIG_FILE
export CONFIG_FILE
export STATUS_DIR
export -f process_single_workspace

PROCS_PER_GPU=4  # 每个GPU并行处理的最大任务数
#PROCS_PER_GPU=8  # 每个GPU并行处理的最大任务数
# --- 4. 任务分配与并行执行 (新逻辑) ---
echo "=========================================================="
echo ">>> 开始扫描并分配任务到各个GPU..."
echo ">>> 每个GPU将并行运行最多 ${PROCS_PER_GPU} 个任务。"
echo "=========================================================="
overall_start_time=$SECONDS

# 获取所有待处理的工作区列表
mapfile -t all_workspaces < <(find "$PARENT_DIR" -maxdepth 1 -mindepth 1 -type d,l)
total_workspaces=${#all_workspaces[@]}
if [ "$total_workspaces" -eq 0 ]; then
    echo "警告: 在 '$PARENT_DIR' 中没有找到任何子目录。"
    exit 0
fi
echo "[INFO] 共发现 ${total_workspaces} 个工作区需要检查和处理。"

# 为每个GPU启动一个独立的“任务处理器”
for (( gpu_id=0; gpu_id<NUM_GPUS; gpu_id++ )); do
    # 这个括号内的代码块会在一个子shell中异步执行
    (
        # 为这个子shell设置可见的GPU
        export CUDA_VISIBLE_DEVICES=$gpu_id
        echo "[INFO] GPU ${gpu_id} 任务处理器已启动。"

        # 遍历所有分配给这个GPU的工作区
        for (( i=gpu_id; i<total_workspaces; i+=NUM_GPUS )); do
            # *** 作业控制逻辑开始 ***
            # 检查当前子shell中的后台作业数量
            # 如果达到上限，则等待
            while [[ $(jobs -p | wc -l) -ge $PROCS_PER_GPU ]]; do
                # 等待任意一个作业完成
                wait -n
            done
            # *** 作业控制逻辑结束 ***

            workspace_path="${all_workspaces[i]}"

            # 将核心处理函数放到后台运行
            process_single_workspace "$workspace_path" &
        done

        # 循环结束后，等待这个GPU上所有剩余的后台作业完成
        echo "[INFO] GPU ${gpu_id}: 所有任务已启动，正在等待处理完成..."
        wait
        echo "[INFO] GPU ${gpu_id}: 所有任务处理完毕。"

    ) & # & 将整个GPU处理器放到后台运行
done

# 等待所有后台的GPU处理器完成任务
echo ">>> 所有任务已分配，等待后台进程执行完毕..."
wait

# --- 5. 最终统计和报告 (这部分不变) ---
overall_duration=$((SECONDS - overall_start_time))
success_count=$(find "$STATUS_DIR" -type f -name "*.success" 2>/dev/null | wc -l)
#fail_count=$((total_workspaces - success_count))

echo "=========================================================="
echo "所有工作区并行处理完毕！"
echo "--- 处理结果摘要 ---"
echo "总计检查工作区: ${total_workspaces}"
echo -e "\033[0;32m成功处理 (或跳过): ${success_count}\033[0m"
#echo -e "\033[0;31m失败: ${fail_count}\033[0m"
printf "脚本总耗时: %d 分 %d 秒\n" $((overall_duration / 60)) $((overall_duration % 60))
echo "=========================================================="