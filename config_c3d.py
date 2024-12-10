# -*- coding: utf-8 -*-
# @Author: XP

from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.COMPLETION3D                        = edict()
__C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH     = './datasets/Completion3D.json'
__C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = './data/c3d/%s/partial/%s/%s.h5'
__C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = './data/c3d/%s/gt/%s/%s.h5'
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.N_RENDERINGS               = 8
__C.DATASETS.SHAPENET.N_POINTS                   = 16384
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '/data/PCN/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '/data/PCN/%s/complete/%s/%s.pcd'

#
# Dataset
#
__C.DATASET                                      = edict()
# Dataset Options: Completion3D, ShapeNet, ShapeNetCars, Completion3DPCCT, JRS
__C.DATASET.TRAIN_DATASET                        = 'JRS'
__C.DATASET.TEST_DATASET                         = 'JRS'

#
# Constants
#
__C.CONST                                        = edict()

__C.CONST.NUM_WORKERS                            = 8
__C.CONST.N_INPUT_POINTS                         = 2048

#
# Directories
#

__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = '/home/wanghao/Projects/PMP-Net-main-WIRE/exp/output'
__C.CONST.DEVICE                                 = '1'
# __C.CONST.WEIGHTS                                = '/home/wanghao/Projects/PMP-Net-main-WIRE/pre-trained/completion3d/ckpt-best-pmpplus.pth'
__C.CONST.WEIGHTS                                = '/home/wanghao/Projects/PMP-Net-main-WIRE/exp/output/checkpoints/2024-12-05T21:28:14.274665/ckpt-best-322.pth'

#
# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.N_SAMPLING_POINTS                    = 2048

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.LAMBDA_CD                              = 1000
__C.TRAIN.LAMBDA_PMD                             = 1e-3   #1e-2
__C.TRAIN.BATCH_SIZE                             = 64     #16
__C.TRAIN.N_EPOCHS                               = 1000
__C.TRAIN.SAVE_FREQ                              = 100
__C.TRAIN.LEARNING_RATE                          = 0.0001  #0.0001
__C.TRAIN.LR_MILESTONES                          = [50, 100, 150, 200, 250]
__C.TRAIN.GAMMA                                  = .5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'

# for JRS
__C.JRS                                          = edict()
__C.JRS.INFERENCE_DATA_PATH                      = '/home/wanghao/Projects/PMP-Net-main-WIRE/data/inference_pts'
# __C.JRS.DATA_PATH                               = '/home/wanghao/Projects/PMP-Net-main-JRS/data/inference_pts/高透明'
__C.JRS.TRAIN_DATA_PATH                          = '/home/wanghao/Projects/PMP-Net-main-WIRE/data/train_pts'
__C.JRS.VAL_DATA_PATH                          = '/home/wanghao/Projects/PMP-Net-main-WIRE/data/val_pts'
__C.JRS.NPOINTS                                  = 2048
