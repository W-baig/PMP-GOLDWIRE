import os
# from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pickle
import random
import math 
from tqdm import tqdm
import re
# 读vipc数据集的dat
# pc_path = "/home/wanghao/Models/EGIInet-main/EGIInet-main/data/ShapeNetViPC-Dataset/ShapeNetViPC-GT/02691156/1a04e3eab45ca15dd86060f189eb133/01.dat" 
# with open(pc_path,'rb') as f:
#     pc = pickle.load(f).astype(np.float32)
#     np.savetxt('02.txt',pc)

import cv2
import numpy as np

class ReadYML:
    def import_yml(file_path):
        # 读取 YAML 文件
        fs = cv2.FileStorage(file_path, cv2.FileStorage_READ)
        # 检查文件是否打开成功
        if not fs.isOpened():
            raise IOError(f"打开文件 {file_path} 失败！")
        # 从文件中读取名为 "height" 的数据节点
        height_image_node = fs.getNode("height")
        if height_image_node.empty():
            raise ValueError(f"文件 {file_path} 中没有找到 'height' 节点！")
        # 将数据节点转换为 NumPy 数组
        height_image = height_image_node.mat()
        # 关闭文件
        fs.release()
        # 检查数据是否为空
        if height_image.size == 0:
            raise ValueError(f"读取高度图像数据失败！")
        
        pts_data = ReadYML.transform_height(height_image)
        pts_data = np.array(pts_data).astype(np.float32)

        model_id = os.path.splitext(os.path.basename(file_path))[0]
        # model_id = re.findall(r'\[(.*?)\]', file_path)
        export_path = os.path.join('/home/wanghao/Projects/PMP-Net-main-JRS/data/pts/高透明', '%s.xyz' % model_id)
        np.savetxt(export_path, pts_data)

        return pts_data

    def transform_height(height_data):
        pts_data = []
        width = height_data.shape[0]
        length = height_data.shape[1]
        
        # transform 2d height img to 3d points
        for w in range(width):
            for l in range(length):
                Z = height_data[w, l]  # 深度值
                # if Z < 0.1 :
                #     continue
                # w_m = w - width/2 
                # l_m = l - length/2
                # if (w_m * w_m + l_m * l_m) >= (width * width / 16):
                # if abs(w_m) >= (width / 4) or abs(l_m) >= (width / 4):
                if 1 :
                    pts_data.append([w/100, l/100, Z]) # 中心居中，等比缩100
                
        return pts_data
        


if __name__ == '__main__':
    file_path = "/home/wanghao/Projects/PMP-Net-main-JRS/data/高反光/heightMats/_a_AlignmentOf3D_2023-08-29_10-26-17_990_2023-10-24_11-28-28_595_ALIGNMENTOF3D [37186, 106197] @ R111_I3_false.yml"

    # 调用函数读取高度图像
    try:
        height_image = ReadYML.import_yml(file_path)

        print(f"读取成功，高度图像大小: {height_image.shape}")
        # np.save("height_image.npy", height_image)
        # cv2.imshow("img",img)
    except Exception as e:
        print(f"发生错误: {e}")