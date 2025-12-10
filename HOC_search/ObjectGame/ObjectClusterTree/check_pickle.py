import pickle
import os
import sys
from .utils import *
from HOC_search.ObjectGame.ClusterProposal import ClusterProposal
from HOC_search.ObjectGame.RotationProposal import RotationProposal
from HOC_search.ObjectGame.CategoryProposal import CategoryProposal
#
# # --- 新增代码：设置项目路径 ---
# # 这部分代码的目的是将HOC_Search项目的根目录添加到Python的搜索路径中
# # 这样Python就能找到像 ObjectGame 这样的自定义模块了
# try:
#     # __file__ 是指当前脚本文件 (check_pickle.py)
#     current_script_path = os.path.dirname(os.path.realpath(__file__))
#     # HOC_Search 目录是当前目录的父目录
#     project_root = os.path.dirname(current_script_path)
#     if project_root not in sys.path:
#         sys.path.append(project_root)
#     print(f"项目根目录 '{project_root}' 已添加到Python路径。")
# except NameError:
#     # 如果在某些交互式环境(如Jupyter)中运行，__file__ 可能不存在
#     # 请手动将 project_root 设置为 HOC_Search 文件夹的绝对路径
#     project_root = '/home/chm/workspace_data1/codehub/2025/HOC_Search'  # <--- 请根据需要修改这里
#     if project_root not in sys.path:
#         sys.path.append(project_root)
#     print(f"项目根目录 '{project_root}' 已手动添加到Python路径。")
# --- 新增代码结束 ---


# --- 请将这里的路径修改为你自己的 tree.pickle 文件的完整路径 ---
pickle_file_path = '/data1/chm/codehub/2025/HOC_Search/data/ShapeNetCore.v2.PC15k_tree/chair/tree.pickle'

print(f"\n正在尝试加载文件: {pickle_file_path}")

try:
    # 'rb' 表示 'read binary' (以二进制读取模式打开)
    with open(pickle_file_path, 'rb') as f:
        # 使用 pickle.load() 来反序列化文件内容
        loaded_data = pickle.load(f)

    print("\n文件加载成功！")
    print("---------------------------------")

    # --- 接下来，检查加载出的数据 ---

    # 1. 查看数据的基本类型 (是字典、列表还是其他?)
    print(f"数据类型: {type(loaded_data)}")

    # 2. 如果是字典，可以看看里面有哪些键 (keys)
    if isinstance(loaded_data, dict):
        print(f"字典的键 (Keys): {list(loaded_data.keys())}")

    # 3. 如果是列表，可以看看它的长度
    if isinstance(loaded_data, list):
        print(f"列表的长度: {len(loaded_data)}")


except Exception as e:
    print(e)
    print("\n文件加载失败！这很可能意味着文件已损坏或不完整。")
    print(f"---------------------------------")
    print(f"捕获到的错误信息: {e}")
    print("\n这个错误信息进一步证实了我们的猜想：这个 tree.pickle 文件是有问题的。")