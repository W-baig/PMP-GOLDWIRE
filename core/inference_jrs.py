# -*- coding: utf-8 -*-
# @Author: XP
import numpy as np
import logging
import os
import torch
import utils.data_loaders
import utils.helpers
import utils.io
import utils.post_process
import utils.edge_extract
from tqdm import tqdm
from models.model import PMPNetPlus as Model

from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation

def inference_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](utils.data_loaders.DatasetSubset.TEST, cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                   batch_size=1,
                                                   num_workers=cfg.CONST.NUM_WORKERS, #8
                                                   collate_fn=utils.data_loaders.collate_fn,
                                                   pin_memory=True,
                                                   shuffle=False)

    # dataset_loader = utils.data_loaders.JRSDataLoader(cfg)
    # test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader, 
    #                                                 batch_size=1,
    #                                                 num_workers=cfg.CONST.NUM_WORKERS,
    #                                                 pin_memory=True,
    #                                                 shuffle=False)

    model = Model(dataset=cfg.DATASET.TRAIN_DATASET)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Load the pretrained model from a checkpoint
    assert 'WEIGHTS' in cfg.CONST and cfg.CONST.WEIGHTS
    logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    # The inference loop
    n_samples = len(test_data_loader)
    t_obj = tqdm(test_data_loader)

    post_processor = utils.post_process.PostProcess(mask_size=(512, 512)) #后处理器

    generate_set = []
    partial_set = []
    pcd3_set = []

    for model_idx, (taxonomy_id, model_id, data, centroid, furthest_distance) in enumerate(t_obj):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():

            

            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            partial = data['partial_cloud']
            


            pcds = model(partial)[0]
            pcd1, pcd2, pcd3 = pcds

            partial = partial.squeeze().cpu().numpy()
            pcd1 = pcd1.squeeze().cpu().numpy()
            pcd2 = pcd2.squeeze().cpu().numpy()
            pcd3 = pcd3.squeeze().cpu().numpy()
        
            partial = partial * 2 * furthest_distance + centroid
            pcd1 = pcd1 * 2 * furthest_distance + centroid
            pcd2 = pcd2 * 2 * furthest_distance + centroid
            pcd3 = pcd3 * 2 * furthest_distance + centroid 

            # # subsampled
            # if pcd3.shape[1] > cfg.JRS.NPOINTS :
            #     pcd3_flipped = pcd3.permute(0, 2, 1).contiguous() # (B, 3, N)
            #     pcd3_flipped = gather_operation(pcd3_flipped, furthest_point_sample(pcd3, 10240)) # (B, 3, NPOINTS)
            #     pcd3 = pcd3_flipped.permute(0, 2, 1).contiguous() # (B, NPOINTS, 3)

            output_folder = os.path.join(cfg.DIR.OUT_PATH, 'benchmark', taxonomy_id)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # output_folder_pcd1 = os.path.join(output_folder, 'pcd1')
            output_folder_pcd3 = os.path.join(output_folder, 'pcd3')
            output_folder_generate = os.path.join(output_folder, 'generate')
            output_folder_partial = os.path.join(output_folder, 'partial')
            if not os.path.exists(output_folder_partial):
                # os.makedirs(output_folder_pcd1)
                # os.makedirs(output_folder_pcd2)
                os.makedirs(output_folder_pcd3)
                os.makedirs(output_folder_generate)
                os.makedirs(output_folder_partial)
            

            filted_pts, _ = post_processor.process(original_pts=partial, generate_pts=pcd3)

            if generate_set == []:
                generate_set = filted_pts
                partial_set = partial
                pcd3_set = pcd3
            else:
                generate_set = np.vstack((generate_set, filted_pts))
                partial_set = np.vstack((partial_set, partial))
                pcd3_set = np.vstack((pcd3_set, pcd3))
            

            # output_file_path = os.path.join(output_folder, 'pcd1', '%s.h5' % model_id)
            # utils.io.IO.put(output_file_path, pcd1.squeeze().cpu().numpy())

            # output_file_path = os.path.join(output_folder, 'pcd2', '%s.h5' % model_id)
            # utils.io.IO.put(output_file_path, pcd2.squeeze().cpu().numpy())

            # output_file_path = os.path.join(output_folder, 'pcd3', '%s.h5' % model_id)
            # utils.io.IO.put(output_file_path, pcd3.squeeze().cpu().numpy())

            output_file_path = os.path.join(output_folder, 'partial', '%s' % model_id)
            np.savetxt(output_file_path, partial)

            # output_file_path = os.path.join(output_folder, 'pcd1', '%s' % model_id)
            # np.savetxt(output_file_path, pcd1)

            output_file_path = os.path.join(output_folder, 'pcd3', '%s' % model_id)
            np.savetxt(output_file_path, pcd3)

            output_file_path = os.path.join(output_folder, 'generate', '%s' % model_id)
            np.savetxt(output_file_path, filted_pts)


            t_obj.set_description('Test[%d/%d] Taxonomy = %s Sample = %s File = %s' %
                         (model_idx + 1, n_samples, taxonomy_id, model_id, output_file_path))
    
    output_folder = os.path.join(cfg.DIR.OUT_PATH, 'benchmark', taxonomy_id)

    output_folder_pcd3 = os.path.join(output_folder, 'pcd3.xyz')
    np.savetxt(output_folder_pcd3, pcd3_set)

    output_folder_generate = os.path.join(output_folder, 'generate.xyz')
    np.savetxt(output_folder_generate, generate_set)

    output_folder_partial = os.path.join(output_folder, 'partial.xyz')
    np.savetxt(output_folder_partial, partial_set)