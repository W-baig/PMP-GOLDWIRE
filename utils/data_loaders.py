# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:21:32
# @Email:  cshzxie@gmail.com

import json
import logging
import numpy as np
import random
import torch.utils.data.dataset
import os
import open3d as o3d
import utils.data_transforms
from enum import Enum, unique
from tqdm import tqdm
from utils.io import IO

from utils.read_yml import ReadYML


label_mapping = {
    3: '03001627',
    6: '04379243',
    5: '04256520',
    1: '02933112',
    4: '03636649',
    2: '02958343',
    0: '02691156',
    7: '04530566'
}

@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def collate_fn(batch):
    pth_id = []
    model_ids = []
    data = {}
    centroid = []
    furthest_distance = []

    for sample in batch:
        pth_id.append(sample[0])
        model_ids.append(sample[1])
        centroid.append(sample[3])
        furthest_distance.append(sample[4])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)  #

    return pth_id, model_ids, data, centroid, furthest_distance



code_mapping = {
    'plane': '02691156',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'lamp': '03636649',
    'couch': '04256520',
    'table': '04379243',
    'watercraft': '04530566',
}

def read_ply(file_path):
    pc = o3d.io.read_point_cloud(file_path)
    ptcloud = np.array(pc.points)
    return ptcloud


class UpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(self.n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, pth_id, data_path, file_list, transforms=None):
        self.options = options
        self.pth_id = pth_id
        self.data_path = data_path
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list['partial'])


    def __getitem__(self, idx):

        file_name = self.file_list['partial'][idx]
        data = {}
        # rand_idx = -1
        
        # if 'n_renderings' in self.options:
        #     rand_idx = random.randint(0, self.options['n_renderings'] - 1) if self.options['shuffle'] else 0
        
        # Load the dataset indexing file

        for f in self.file_list :
            if f == 'partial':
                data['partial_cloud'] = np.loadtxt(os.path.join(self.data_path, 'partial', file_name))
            if f == 'gt':
                data['gtcloud'] = np.loadtxt(os.path.join(self.data_path, 'gt', file_name))
        # height_data = ReadYML.import_yml(os.path.join(self.data_path, sample))
        # pts_data = ReadYML.transform_height(height_data)
        # pts_data = np.array(pts_data).astype(np.float32)


        # for ri in self.options['required_items']:
        #     file_path = sample['%s_path' % ri]
        #     if type(file_path) == list:
        #         file_path = file_path[rand_idx]
        #     # print(file_path)
        #     data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data, centroid, furthest_distance = self.transforms(data)

        pth_id = self.pth_id

        return pth_id, file_name, data, centroid, furthest_distance


class JRSDataLoader(object):
    def __init__(self, subset, cfg):

        if subset == DatasetSubset.TEST:
            self.data_path = cfg.JRS.INFERENCE_DATA_PATH
        elif subset == DatasetSubset.TRAIN:
            self.data_path = cfg.JRS.TRAIN_DATA_PATH
        elif subset == DatasetSubset.VAL:
            self.data_path = cfg.JRS.VAL_DATA_PATH


        self.cfg = cfg
        self.pth_id = cfg.CONST.WEIGHTS if 'WEIGHTS' in cfg.CONST and cfg.CONST.WEIGHTS else "0/0.0"
        self.npoints = cfg.JRS.NPOINTS
        self.files = os.listdir(self.data_path)
        

    def get_dataset(self, subset):
        pth_id = self.pth_id.split('.')[-2].split('/')[0]
        data_path = self.data_path
        files = self.files
        file_list = {}
        for f in files:
            file_list[f] = os.listdir(os.path.join(self.data_path, f))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud'] if subset == DatasetSubset.TEST else ['partial_cloud', 'gtcloud']
        return Dataset({
            'required_items': required_items,
            'shuffle': subset == DatasetSubset.TRAIN}
        , pth_id, data_path, file_list, transforms)        

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'Normalize',
                'objects': ['partial_cloud', 'gtcloud']
            },{
                'callback': 'UpSamplePoints',
                'parameters': {
                'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud', 'gtcloud']
            },
            # {
            #     'callback': 'RandomClipPoints',
            #     'parameters': { },
            #     'objects': ['partial_cloud']
            # },
            {
                'callback': 'RandomRotatePoints',
                'objects': ['partial_cloud', 'gtcloud']
            },{
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        elif subset == DatasetSubset.VAL:
            return utils.data_transforms.Compose([{
                'callback': 'Normalize',
                'objects': ['partial_cloud', 'gtcloud']
            },{
                'callback': 'UpSamplePoints',
                'parameters': {
                'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud', 'gtcloud']
            },
            # {
            #     'callback': 'RandomClipPoints',
            #     'parameters': { },
            #     'objects': ['partial_cloud']
            # },
            {
                'callback': 'RandomRotatePoints',
                'objects': ['partial_cloud', 'gtcloud']
            },
            {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                    'callback': 'Normalize',
                    'objects': ['partial_cloud']
                },
                {
                    'callback': 'UpSamplePoints',
                    'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                    },
                    'objects': ['partial_cloud']
                },
                # {
                #     'callback': 'RandomClipPoints',
                #     'parameters': { },
                #     'objects': ['partial_cloud']
                # },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial_cloud']
                }])
# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {

    'JRS': JRSDataLoader

}  # yapf: disable

