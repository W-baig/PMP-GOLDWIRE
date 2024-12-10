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
    taxonomy_ids = []
    model_ids = []
    data = {}
    centroid = []
    furthest_distance = []

    for sample in batch:
        taxonomy_ids.append(sample[0])
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

    return taxonomy_ids, model_ids, data, centroid, furthest_distance



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


class MyShapeNetDataSet(torch.utils.data.dataset.Dataset):
    def __init__(self, root='/data1/xp/PCN', phase='train', categories=None):
        assert phase in {'train', 'val', 'test'}
        self.phase = phase
        base_dir = os.path.join(root, phase)
        if categories is None:
            self.taxomony_ids = list(code_mapping.values())
        else:
            taxomony_ids = []
            for c in categories:
                taxomony_ids.append(code_mapping[c])
            self.taxomony_ids = taxomony_ids

        all_taxomony_ids = []
        all_model_ids = []
        all_pcds_partial = []
        all_pcds_gt = []

        for t_id in self.taxomony_ids:
            gt_dir = os.path.join(base_dir, 'complete', t_id)
            partial_dir = os.path.join(base_dir, 'partial', t_id)
            model_ids = os.listdir(partial_dir)
            all_taxomony_ids.extend([t_id for i in range(len(model_ids))])
            all_model_ids.extend(model_ids)
            all_pcds_gt.extend([os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir))])
            all_pcds_partial.extend([os.path.join(partial_dir, f) for f in sorted(os.listdir(partial_dir))])

        self.taxomony_ids = all_taxomony_ids
        self.model_ids = all_model_ids
        self.path_partial = all_pcds_partial
        self.path_gt = all_pcds_gt
        self.LEN = len(self.model_ids)
        self.transform = UpSamplePoints({'n_points': 2048})

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, index):
        if self.phase == 'test':
            partial = read_ply(self.path_partial[index]).astype(np.float32)
        else:
            idx_partial = random.randint(0, 7)
            partial = read_ply(os.path.join(self.path_partial[index], '0{}.pcd'.format(idx_partial))).astype(np.float32)
        partial = self.transform(partial)
        gt = read_ply(self.path_gt[index]).astype(np.float32)
        idx_random_complete = random.randint(0, self.LEN - 1)
        random_complete = read_ply(self.path_gt[idx_random_complete]).astype(np.float32)
        data = {
            'X': torch.from_numpy(partial).float(),
            'Y': torch.from_numpy(random_complete).float(),
            'X_GT': torch.from_numpy(gt).float()
        }
        return self.taxomony_ids[index], self.model_ids[index], data


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, taxonomy_ids, data_path, file_list, transforms=None):
        self.options = options
        self.taxonomy_ids = taxonomy_ids
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

        taxonomy_ids = self.taxonomy_ids

        return taxonomy_ids, file_name, data, centroid, furthest_distance


class ShapeNetDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):

                if subset == 'test':

                    gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                    file_list.append({'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': gt_path.replace('complete', 'partial'),
                    'gtcloud_path': gt_path})
                else:
                    file_list.append({
                        'taxonomy_id':
                            dc['taxonomy_id'],
                        'model_id':
                            s,
                        'partial_cloud_path': [
                            cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s, i)
                            for i in range(n_renderings)
                        ],
                        'gtcloud_path':
                            cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                    })

                    '''
                    gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                    file_list.extend([{
                        'taxonomy_id': dc['taxonomy_id'],
                        'model_id': s,
                        'partial_cloud_path': cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (
                        subset, dc['taxonomy_id'], s, i),
                        'gtcloud_path': gt_path
                    } for i in range(n_renderings)])
                    '''

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class ShapeNetCarsDataLoader(ShapeNetDataLoader):
    def __init__(self, cfg):
        super(ShapeNetCarsDataLoader, self).__init__(cfg)

        # Remove other categories except cars
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']


class Completion3DDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud'] if subset == DatasetSubset.TEST else ['partial_cloud', 'gtcloud']
        return Dataset({
            'required_items': required_items,
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial_cloud', 'gtcloud']
                }])
        elif subset == DatasetSubset.VAL:
            return utils.data_transforms.Compose([{
                'callback': 'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_cloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path':
                    cfg.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                    'gtcloud_path':
                    cfg.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list

class JRSDataLoader(object):
    def __init__(self, subset, cfg):

        if subset == DatasetSubset.TEST:
            self.data_path = cfg.JRS.INFERENCE_DATA_PATH
        elif subset == DatasetSubset.TRAIN:
            self.data_path = cfg.JRS.TRAIN_DATA_PATH
        elif subset == DatasetSubset.VAL:
            self.data_path = cfg.JRS.VAL_DATA_PATH


        self.cfg = cfg
        self.taxonomy_ids = cfg.CONST.WEIGHTS if 'WEIGHTS' in cfg.CONST and cfg.CONST.WEIGHTS else "0/0.0"
        self.npoints = cfg.JRS.NPOINTS
        self.files = os.listdir(self.data_path)
        

    def get_dataset(self, subset):
        taxonomy_ids = self.taxonomy_ids.split('.')[-2].split('/')[0]
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
        , taxonomy_ids, data_path, file_list, transforms)        

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
                # {
                #     'callback': 'RandomRotatePoints',
                #     'objects': ['partial_cloud']
                # },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial_cloud']
                }])
            



class Completion3DPCCTDataLoader(Completion3DDataLoader):
    """
    Dataset Completion3D containing only plane, car, chair, table
    """
    def __init__(self, cfg):
        super(Completion3DPCCTDataLoader, self).__init__(cfg)

        # Remove other categories except couch, chairs, car, lamps
        cat_set = {'02691156', '03001627', '02958343', '04379243'} # plane, chair, car, table
        # cat_set = {'04256520', '03001627', '02958343', '03636649'}
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] in cat_set]


class KittiDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.KITTI.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud', 'bounding_box']

        return Dataset({'required_items': required_items, 'shuffle': False}, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        return utils.data_transforms.Compose([{
            'callback': 'NormalizeObjectPose',
            'parameters': {
                'input_keys': {
                    'ptcloud': 'partial_cloud',
                    'bbox': 'bounding_box'
                }
            },
            'objects': ['partial_cloud', 'bounding_box']
        }, {
            'callback': 'RandomSamplePoints',
            'parameters': {
                'n_points': cfg.CONST.N_INPUT_POINTS
            },
            'objects': ['partial_cloud']
        }, {
            'callback': 'ToTensor',
            'objects': ['partial_cloud', 'bounding_box']
        }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': cfg.DATASETS.KITTI.PARTIAL_POINTS_PATH % s,
                    'bounding_box_path': cfg.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH % s,
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'Completion3D': Completion3DDataLoader,
    'JRS': JRSDataLoader,
    'Completion3DPCCT': Completion3DPCCTDataLoader,
    'ShapeNet': ShapeNetDataLoader,
    'ShapeNetCars': ShapeNetCarsDataLoader,
    'KITTI': KittiDataLoader
}  # yapf: disable

