import open3d as o3d

file1 = "/data1/chm/codehub/2025/HOC_Search/output/scene0549_01/scene0549_01_mesh_full_bg.ply"
file2 = "/data1/chm/codehub/2025/HOC_Search/output/scene0549_01/scene0549_01_cad_retrieval.ply"

# 加载第一个 .ply 文件
mesh1 = o3d.io.read_triangle_mesh(file1)

# 加载第二个 .ply 文件
mesh2 = o3d.io.read_triangle_mesh(file2)

# 确保两个网格都正确加载
if mesh1.is_empty() or mesh2.is_empty():
    print("其中一个网格为空，请检查文件是否有效。")
else:
    # 合并两个网格
    combined_mesh = mesh1 + mesh2

    # 保存合并后的网格到新文件
    o3d.io.write_triangle_mesh("/data1/chm/codehub/2025/HOC_Search/output/scene0549_01/combined.ply", combined_mesh)

    print("网格已成功合并并保存为 'combined.ply'")
