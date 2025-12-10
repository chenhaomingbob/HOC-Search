#!/bin/bash

# ### 使用说明 ###
#
# 1. 将此脚本放置在你的项目根目录下 (与 preprocess_ArkitScene.py 的父目录同级)。
# 2. 在终端中，赋予此脚本执行权限:
#    chmod +x batch_process.sh
# 3. 执行脚本，并提供工作区根目录的路径:
#    ./batch_process.sh /path/to/your/workspace
#
# 例如:
# 如果你的场景数据结构如下:
# /data/ArkitScenes/
# ├── 40448135
# │   ├── mesh.ply
# │   ├── pose/
# │   └── intrinsic/
# ├── 40448152
# │   ├── mesh.ply
# │   ├── pose/
# │   └── intrinsic/
# ...
#
# 你应该这样运行脚本:
# ./batch_process.sh /data/ArkitScenes

# ### 配置 ###

# 检查是否提供了工作区根目录
if [ -z "$1" ]; then
  echo "错误: 请提供工作区根目录的路径。"
  echo "用法: $0 <workspace_root_directory>"
  exit 1
fi

# 工作区根目录 (从第一个命令行参数获取)
WORKSPACE_ROOT="$1"

# Python 脚本的路径
# 假设此 bash 脚本与 preprocess_ArkitScene.py 在同一目录下
#PYTHON_SCRIPT_DIR=$(dirname "$0") # 获取当前脚本所在的目录
#PYTHON_SCRIPT_NAME="preprocess_ArkitScene.py"
#PYTHON_SCRIPT_PATH="${PYTHON_SCRIPT_DIR}/${PYTHON_SCRIPT_NAME}"
PYTHON_SCRIPT_PATH="scene_preprocessing/preprocess_ArkitScene.py"

# 输出文件夹名称 (与你的 Python 脚本中的默认值保持一致)
OUTPUT_FOLDER="intermediate/HOC_Search"

# ### 脚本主体 ###

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "错误: Python 脚本未找到于: ${PYTHON_SCRIPT_PATH}"
    exit 1
fi

# 遍历 $WORKSPACE_ROOT 下的每一个项目 (仅限目录)
for scene_dir in "$WORKSPACE_ROOT"/*/; do
    # 检查 scene_dir 是否是一个存在的目录
    if [ ! -d "$scene_dir" ]; then
        continue # 如果不是目录, 则跳过
    fi

#    echo "${scene_dir}intermediate/HOC_Search/bg.ply"
    if [ -f "${scene_dir}intermediate/HOC_Search/bg.ply" ]; then
        echo "发现 'bg.ply', 跳过场景: ${scene_dir}"
        continue
    fi

    # 判断当前子目录中是否直接包含 mesh.ply 文件
    if [ -f "${scene_dir}mesh.ply" ]; then
        # 如果文件存在, 则处理这个场景目录
        echo "=========================================================="
        # a=${scene_dir%*/} # 这行可以去除路径末尾的'/'
        # echo "发现 'mesh.ply', 正在处理场景: ${a##*/}"
        echo "发现 'mesh.ply', 正在处理场景: ${scene_dir}"
        echo "=========================================================="



        python3 "$PYTHON_SCRIPT_PATH" \
            --workspace "$scene_dir" \
            --output "$OUTPUT_FOLDER"

        if [ $? -ne 0 ]; then
            echo "警告: 处理场景 ${scene_dir} 时发生错误。"
        else
            echo "场景 ${scene_dir} 处理完成。"
        fi
        echo
    fi
done

echo "=========================================================="
echo "所有场景处理完毕。"
echo "=========================================================="