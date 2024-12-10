import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
import open3d as o3d

def generate_defect(points):

    # 计算所有点对之间的距离
    distances = pdist(points)  # pairwise distances
    dist_matrix = squareform(distances)  # 转换为对称距离矩阵

    # 找到距离最大的两个点的索引
    i, j = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    point_start, point_end = points[i], points[j]

    ex_2_points = np.vstack([point_start, point_end])
    # output_path = "/home/wanghao/Projects/goldwire/get_ply_from_tiff_and_txt/2_4/super_8_0_pointCloud/ex_2_points.xyz"  # 指定保存路径
    # np.savetxt(output_path, ex_2_points)

    num_segments = np.random.randint(3, 10) # 分割数量
    segment_ratio = np.linspace(0, 1, num_segments + 1)  # 分割比例
    segments = []

    # 计算主轴方向的单位向量
    axis_vector = (point_end - point_start) / np.linalg.norm(point_end - point_start)

    # 对每个分割比例生成切片
    for k in range(num_segments):
        start_ratio, end_ratio = segment_ratio[k], segment_ratio[k + 1]
        segment_start = point_start + start_ratio * (point_end - point_start)
        segment_end = point_start + end_ratio * (point_end - point_start)

        # 获取在此分割区间内的点
        segment_mask = ((points - segment_start) @ axis_vector >= 0) & ((points - segment_end) @ axis_vector < 0)
        segment_points = points[segment_mask]
        segments.append(segment_points)


    # 随机选择要移除的段索引
    Idx_to_remove = np.random.randint(1, num_segments -1)  # 缺损段数量，可根据需要调整
    # 生成缺损后的点云
    occluded_points = np.vstack([seg for idx, seg in enumerate(segments) if idx != Idx_to_remove])

    print("分段：", num_segments, "截断：", Idx_to_remove)

    return occluded_points


if __name__ == '__main__':
    file_path = "/home/wanghao/Projects/PMP-Net-main-WIRE/data/upper_intact/32_top"

    files = os.listdir(file_path)

    min_partial_pts_num = 1e4
    min_gt_pts_num = 1e4


    #转ply到xyz
    for data in files:
            data_path = os.path.join(file_path, data) #文件路径
            point_cloud = o3d.io.read_point_cloud(data_path)
            points = np.asarray(point_cloud.points)
            offset = [0, 0, -0.32]
            points = points + offset
            output_path = "/home/wanghao/Projects/PMP-Net-main-WIRE/data/inference_pts/partial"
            data_name = os.path.splitext(data)[0] #文件名不含后缀
            output_data_name = f"{data_name}.xyz"
            np.savetxt(os.path.join(output_path, output_data_name), points)    


    #生成缺陷
    # for f in files:
    #     if f == 'n_use': continue #忽略不用的点
    #     datas = os.listdir(os.path.join(file_path, f))
    #     for data in datas:
    #         data_path = os.path.join(file_path, f, data) #文件路径
    #         point_cloud = o3d.io.read_point_cloud(data_path)
    #         points = np.asarray(point_cloud.points)
    #         cur_pts_num = points.shape[0]
    #         if cur_pts_num < min_gt_pts_num : min_gt_pts_num = cur_pts_num

            
    #         try:
    #             for i in range(3):
    #                 occluded_points = generate_defect(points)
    #                 cur_pts_num = occluded_points.shape[0]
    #                 if cur_pts_num < min_partial_pts_num : min_partial_pts_num = cur_pts_num
    #                 output_path = "/home/wanghao/Projects/PMP-Net-main-WIRE/data/train_pts"
    #                 data_name = os.path.splitext(data)[0] #文件名不含后缀
    #                 output_data_name = f"{data_name}_{i}.xyz"
    #                 np.savetxt(os.path.join(output_path, 'partial', output_data_name), occluded_points)
    #                 np.savetxt(os.path.join(output_path, 'gt', output_data_name), points)                    


    #             print(f"success {data_name}")
    #             # np.save("height_image.npy", height_image)
    #             # cv2.imshow("img",img)
    #         except Exception as e:
    #             print(f"error: {e}")
    
    # print("min partial nums", min_partial_pts_num)  #669
    # print("min gt nums", min_gt_pts_num)  #1040