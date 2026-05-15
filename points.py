import open3d as o3d


def count_points_in_ply(ply_file):
    # 使用 open3d 读取 PLY 文件
    point_cloud = o3d.io.read_point_cloud(ply_file)

    # 获取点云数量
    num_points = len(point_cloud.points)

    print(f"Number of points in {ply_file}: {num_points}")
    return num_points


# 测试
ply_file = "/media/zgy/data/LXT/ex-model/FSGS-main/output/bicycle/point_cloud/iteration_10000/point_cloud.ply"
count_points_in_ply(ply_file)
