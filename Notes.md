
# 使用流程

1.通过MASK3D实例分割的结果来生成所需的HOC-Search所需要的.pkl文件
python preprocess_ArkitScene.py --workspace /home/chm/datasets/arkitscenes/labelmaker/47333462

2.进行实际的检索 
python CAD_retrieval_HOC_search_ScanNetpp.py --config ArkitScene_HOC_Search.ini

# 批量脚本
python preprocess_ArkitScene.py --workspac/data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v2/41048169
./scripts/batch_arkitscene_process.sh  /data1/chm/datasets/arkitscenes/LabelMaker/mini_data
修改对应的ArkitScene_HOC_Search.ini的dataset_base_path和data_folder

CUDA_VISIBLE_DEVICES=1 ./scripts/batch_arkitscene_process.sh  /data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v3
CUDA_VISIBLE_DEVICES=1 python CAD_retrieval_HOC_search_ArkitScene.py --config ArkitScene_HOC_Search.ini


## 多进程并行执行CAD检索

# Step1
bash ./scripts/batch_arkitscene_process.sh  /data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v3
bash ./scripts/batch_arkitscene_process.sh  /data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v4

bash ./scripts/batch_arkitscene_process.sh  /data1/chm/datasets/arkitscenes/LabelMaker/Training
# Step2
## 先处理一下640x480的
bash scripts/batch_arkitscene_cad_retrieval.sh /data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v3
## 再启动处理480x640的

# 请手动修改相关的参数
```shell
config/ArkitScene_HOC_Search.ini  L13/L4
HOC_search/CAD_retrieval_HOC_search_ArkitScene.py L115/L116
HOC_search/CAD_retrieval_HOC_search_ArkitScene.py L171
```
bash scripts/batch_arkitscene_cad_retrieval.sh /data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v3



bash scripts/batch_arkitscene_cad_retrieval.sh /data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v4 ArkitScene_HOC_Search_v4_h640_w480.ini
bash scripts/batch_arkitscene_cad_retrieval.sh /data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v4 ArkitScene_HOC_Search_v4_h480_w640.ini

bash scripts/batch_arkitscene_cad_retrieval.sh /data1/chm/datasets/arkitscenes/LabelMaker/Training ArkitScene_HOC_Search_h640_w480.ini
bash scripts/batch_arkitscene_cad_retrieval.sh /data1/chm/datasets/arkitscenes/LabelMaker/Training ArkitScene_HOC_Search_h480_w640.ini
