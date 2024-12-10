import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
import open3d as o3d

class PostProcess:
    def __init__(self, mask_size=(512, 512)):
        self.mask_size = mask_size
    
    def project_to_xy(self, original_pts):
        return original_pts[:, :2]
    
    def compute_average_distance(self, original_pts):

        points_xy = original_pts[:, :2]
        kdtree = cKDTree(points_xy)
        distances, _ = kdtree.query(points_xy, k=2)  # k=2 因为查询的是自己的邻居（距离是 0）
        nearest_distances = distances[:, 1]
        avg_distance = np.mean(nearest_distances)
        return avg_distance
    
    def compute_scale(self, original_pts):

        min_vals = np.min(original_pts[:, :2], axis=0)  # 获取 XY 轴上的最小值
        max_vals = np.max(original_pts[:, :2], axis=0)  # 获取 XY 轴上的最大值

        width = max_vals[0] - min_vals[0]
        height = max_vals[1] - min_vals[1]

        target_width, target_height = self.mask_size

        scale_x = target_width / width
        scale_y = target_height / height
        scale = min(scale_x, scale_y)  # 保持图像比例
        
        return scale
    
    def create_mask(self, original_pts, generate_pts, avg_distance):

        xy_pts = self.project_to_xy(original_pts)
        scale = self.compute_scale(original_pts) * 0.7
        img_pts = np.round(xy_pts * scale)
        centroid = np.mean(img_pts, axis=0)
        img_pts = (img_pts - centroid + (256, 256)).astype(int) #centor
        mask = np.zeros(self.mask_size)
        radius = avg_distance * scale * 4
        for pt in img_pts:
            x, y = pt
            if 0 <= x < self.mask_size[1] and 0 <= y < self.mask_size[0]:
                # 计算每个点周围的区域，并填充 mask
                y_min = max(0, y - int(radius))
                y_max = min(self.mask_size[0], y + int(radius))
                x_min = max(0, x - int(radius))
                x_max = min(self.mask_size[1], x + int(radius))
                
                # 用一个简单的圆形范围来填充 mask
                for i in range(y_min, y_max):
                    for j in range(x_min, x_max):
                        if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2:
                            mask[i, j] = 1  # 点周围的区域标记为 1
            else:
                print("out range")

        generate_pts = generate_pts * scale
        centroid = np.mean(generate_pts, axis=0)
        generate_pts = generate_pts - centroid + (256, 256, 0)#centor

        x_coords = generate_pts[:, 0]
        y_coords = generate_pts[:, 1]

        # 计算 mask 的范围
        x_min = 0
        y_min = 0
        x_max = mask.shape[1]
        y_max = mask.shape[0]

        # 保证点坐标在 mask 的范围内，超出范围的点不会参与过滤
        valid_mask = (x_coords >= x_min) & (x_coords < x_max) & (y_coords >= y_min) & (y_coords < y_max)

        # 将有效的点的索引转换为二维索引 (x, y)，并检查其是否在 mask 上
        mask_x = np.clip(np.floor(x_coords[valid_mask]).astype(int), 0, mask.shape[1] - 1)
        mask_y = np.clip(np.floor(y_coords[valid_mask]).astype(int), 0, mask.shape[0] - 1)

        # 根据 mask 值过滤掉位于 mask 中 (True 区域) 的点
        filter_mask = mask[mask_y, mask_x] == False
        filted_pts = generate_pts[valid_mask]
        filted_pts = filted_pts[filter_mask]

        filted_pts = filted_pts + centroid - (256, 256, 0)
        filted_pts = filted_pts / scale

        return filted_pts, mask
    
    def process(self, original_pts, generate_pts):

        original_pts = np.unique(original_pts, axis=0)

        # min_vals = np.min(original_pts[:, :2], axis=0)  # 获取 XY 轴上的最小值
        # shift_x = -min_vals[0]
        # shift_y = -min_vals[1]
        # original_pts[:, 0] += shift_x
        # original_pts[:, 1] += shift_y
        
        avg_distance = self.compute_average_distance(original_pts)
        filted_pts, mask = self.create_mask(original_pts, generate_pts, avg_distance)

        return filted_pts, mask


# 使用示例
# 假设点云数据
data_path = "/home/wanghao/Projects/PMP-Net-main-WIRE/exp/output/benchmark/389526/partial/baseView15_bottom.xyz"#baseView0_bottom.xyz
original_pts = o3d.io.read_point_cloud(data_path)
original_pts = np.asarray(original_pts.points)

data_path = "/home/wanghao/Projects/PMP-Net-main-WIRE/exp/output/benchmark/389526/pcd3/baseView15_bottom.xyz"#baseView0_bottom.xyz
generate_pts = o3d.io.read_point_cloud(data_path)
generate_pts = np.asarray(generate_pts.points)


# 创建点云到 mask 的转换器
processor = PostProcess(mask_size=(512, 512))

# 生成 mask
filted_pts, mask = processor.process(original_pts, generate_pts)

# 显示结果
np.savetxt("/home/wanghao/Projects/PMP-Net-main-WIRE/utils/filted_pts.xyz", filted_pts)
plt.imsave("/home/wanghao/Projects/PMP-Net-main-WIRE/utils/mask.png", mask, cmap='gray')
