#!/usr/bin/env python

'''
Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Extracts Patch-NetVLAD local and NetVLAD global features from a given directory of images.

Configuration settings are stored in configs folder, with compute heavy performance or light-weight alternatives
available.

Features are saved into a nominated output directory, with one file per image per patch size.

Code is dynamic and can be configured with essentially *any* number of patch sizes, by editing the config files.
'''


import argparse
import configparser
import os
from os.path import join, isfile

from tqdm.auto import tqdm
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import pandas as pd

import file_config
from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
from patchnetvlad.tools.patch_matcher import PatchMatcher
from patchnetvlad.tools.datasets import input_transform
from patchnetvlad.models.local_matcher import calc_keypoint_centers_from_patches as calc_keypoint_centers_from_patches
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR


image_path = '/media/yushichen/LENOVO_USB_HDD/IPIN_res/db'

def apply_patch_weights(input_scores, num_patches, patch_weights):
    output_score = 0
    if len(patch_weights) != num_patches:
        raise ValueError('The number of patch weights must equal the number of patches used')
    for i in range(num_patches):
        output_score = output_score + (patch_weights[i] * input_scores[i])
    return output_score


def plot_two(cv_im_one, cv_im_two, inlier_keypoints_one, inlier_keypoints_two, plot_save_path):

    kp_all1 = []
    kp_all2 = []
    matches_all = []
    for this_inlier_keypoints_one, this_inlier_keypoints_two in zip(inlier_keypoints_one, inlier_keypoints_two):
        for i in range(this_inlier_keypoints_one.shape[0]):
            kp_all1.append(cv2.KeyPoint(this_inlier_keypoints_one[i, 0].astype(float), this_inlier_keypoints_one[i, 1].astype(float), 1, -1, 0, 0, -1))
            kp_all2.append(cv2.KeyPoint(this_inlier_keypoints_two[i, 0].astype(float), this_inlier_keypoints_two[i, 1].astype(float), 1, -1, 0, 0, -1))
            matches_all.append(cv2.DMatch(i, i, 0))

    im_allpatch_matches = cv2.drawMatches(cv_im_one, kp_all1, cv_im_two, kp_all2,
                                          matches_all, None, matchColor=(0, 255, 0), flags=2)
    if plot_save_path is None:
        cv2.imshow('frame', im_allpatch_matches)
    else:
        im_allpatch_matches = cv2.cvtColor(im_allpatch_matches, cv2.COLOR_BGR2RGB)

        plt.imshow(im_allpatch_matches)
        # plt.show()
        plt.axis('off')
        filename = join(plot_save_path, 'patchMatchings.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


def match_two(model, device, config, im_one, feature_dir):

    df = pd.read_csv('/media/yushichen/LENOVO_USB_HDD/dataset/tum_sfm/gt_kitti.txt', sep=' ', header=None)

    pool_size = int(config['global_params']['num_pcs'])

    model.eval()
    print('check model device: ', next(model.parameters()).device)

    it = input_transform((int(config['feature_extract']['imageresizeH']), int(config['feature_extract']['imageresizeW'])))

    im_one_pil = Image.fromarray(cv2.cvtColor(im_one, cv2.COLOR_BGR2RGB))
    # im_two_pil = Image.fromarray(cv2.cvtColor(im_two, cv2.COLOR_BGR2RGB))

    im_one_pil = it(im_one_pil).unsqueeze(0)
    # im_two_pil = it(im_two_pil).unsqueeze(0)

    # input_data = torch.cat((im_one_pil.to(device), im_two_pil.to(device)), 0)
    im_one_pil = im_one_pil.to(device)
    input_data = im_one_pil

    tqdm.write('====> Extracting Features')
    time1 = time.time()
    with torch.no_grad():
        image_encoding = model.encoder(input_data)
        temp = time.time()
        vlad_local, _ = model.pool(image_encoding)
        # global_feats = get_pca_encoding(model, vlad_global).cpu().numpy()

        time2 = time.time()

        local_feats_one = []
        local_feats_two = []
        for this_iter, this_local in enumerate(vlad_local):
            this_local_feats = get_pca_encoding(model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))). \
                reshape(this_local.size(2), this_local.size(0), pool_size).permute(1, 2, 0)
            local_feats_one.append(torch.transpose(this_local_feats[0, :, :], 0, 1))
            # local_feats_two.append(this_local_feats[1, :, :])


    tqdm.write('====> Calculating Keypoint Positions')
    patch_sizes = [int(s) for s in config['global_params']['patch_sizes'].split(",")]
    strides = [int(s) for s in config['global_params']['strides'].split(",")]
    patch_weights = np.array(config['feature_match']['patchWeights2Use'].split(",")).astype(float)

    all_keypoints = []
    all_indices = []

    tqdm.write('====> Matching Local Features')
    for patch_size, stride in zip(patch_sizes, strides):
        # we currently only provide support for square patches, but this can be easily modified for future works
        keypoints, indices = calc_keypoint_centers_from_patches(config['feature_match'], patch_size, patch_size, stride, stride)
        all_keypoints.append(keypoints)
        all_indices.append(indices)

    matcher = PatchMatcher(config['feature_match']['matcher'], patch_sizes, strides, all_keypoints,
                           all_indices)
    # matching query with db feature
    # feature_dir = '/media/yushichen/LENOVO_USB_HDD/dataset/tum_sfm/output_feature'
    db_feature_list = os.listdir(feature_dir)
    db_feature_list.sort()
    place1 = time.time()
    for i in range(0, len(db_feature_list), 20):
        if db_feature_list[i] == 'globalfeats.npy':
            continue
        local_feats_two_np = np.load(os.path.join(feature_dir, db_feature_list[i]))
        local_feats_two = [torch.from_numpy(local_feats_two_np).to(device)]
        scores, inlier_keypoints_one, inlier_keypoints_two = matcher.match(local_feats_one, local_feats_two)
        score = -apply_patch_weights(scores, len(patch_sizes), patch_weights)
        file_name = db_feature_list[i].lstrip('patchfeats_psize5_').rstrip('npy') + 'png'
        # if score >= 1000:
        #     print(df.iloc[i])
            # label = df.iloc[i]
            # ref_pose = np.array([[label[0], label[1], label[2], label[3]],
            #                      [label[4], label[5], label[6], label[7]],
            #                      [label[8], label[9], label[10], label[11]],
            #                      [0., 0., 0., 1.]])
            # im_two = cv2.imread(os.path.join(image_path, file_name), -1)
            # trans = calculate_relative_pose(im_one, im_two)
            # abs_pose = np.matmul(ref_pose, np.linalg.inv(trans))
            # print(abs_pose)
        print(f"Similarity score between the two images is: {score:.5f}. Larger scores indicate better matches.", file_name)
    place2 = time.time()

    time3 = time.time()
    # print('encoder time: ', temp - time1)
    print('extracting time: ', time2 - time1)
    print('matching time: ', place2 - place1)
    # print('total time: ', time3 - time1)


    # if config['feature_match']['matcher'] == 'RANSAC':
    #     if plot_save_path is not None:
    #         tqdm.write('====> Plotting Local Features and save them to ' + str(join(plot_save_path, 'patchMatchings.png')))
    #
    #     # using cv2 for their in-built keypoint correspondence plotting tools
    #     cv_im_one = cv2.resize(im_one, (int(config['feature_extract']['imageresizeW']), int(config['feature_extract']['imageresizeH'])))
    #     cv_im_two = cv2.resize(im_two, (int(config['feature_extract']['imageresizeW']), int(config['feature_extract']['imageresizeH'])))
    #     # cv2 resize slightly different from torch, but for visualisation only not a big problem
    #
    #     plot_two(cv_im_one, cv_im_two, inlier_keypoints_one, inlier_keypoints_two, plot_save_path)


def main():
    parser = argparse.ArgumentParser(description='Patch-NetVLAD-Match-Two')
    parser.add_argument('--config_path', type=str, default='./patchnetvlad/configs/speed.ini',
                        help='File name (with extension) to an ini file that stores most of the configuration data for patch-netvlad')
    parser.add_argument('--first_im_path', type=str, default='/media/yushichen/LENOVO_USB_HDD/IPIN_res/site_A/seq_1/query/frame_000962.png',
                        help='Full path (with extension) to an image file')
    parser.add_argument('--feature_dir', type=str, default='/media/yushichen/LENOVO_USB_HDD/IPIN_res/site_A/seq_5/output_feature',
                        help='Full path (with extension) to another image file')
    # parser.add_argument('--plot_save_path', type=str, default=join(PATCHNETVLAD_ROOT_DIR, 'results'),
    #                     help='Path plus optional prefix pointing to a location to save the output matching plot')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')

    opt = parser.parse_args()
    print(opt)

    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    encoder_dim, encoder = get_backend()

    # must resume to do extraction
    resume_ckpt = config['global_params']['resumePath'] + config['global_params']['num_pcs'] + '.pth.tar'

    # backup: try whether resume_ckpt is relative to script path
    # if not isfile(resume_ckpt):
    #     resume_ckpt = join(PATCHNETVLAD_ROOT_DIR, resume_ckpt)
    #     if not isfile(resume_ckpt):
    #         from download_models import download_all_models
    #         download_all_models(ask_for_permission=True)

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] == int(config['global_params']['num_pcs'])
        config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

        model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=True)

        if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
            model.encoder = nn.DataParallel(model.encoder)
            # if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)

        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        print("=> loaded checkpoint '{}'".format(resume_ckpt, ))
    else:
        raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_ckpt))

    im_one = cv2.imread(opt.first_im_path, -1)
    if im_one is None:
        raise FileNotFoundError(opt.first_im_path + " does not exist")
    # im_two = cv2.imread(opt.second_im_path, -1)
    # if im_two is None:
    #     raise FileNotFoundError(opt.second_im_path + " does not exist")

    match_two(model, device, config, im_one, opt.feature_dir)

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs
    print('Done')


def relocalize(im_path, feature_dir, gt_dir):
    print('Start relocalization!!!')
    # configfile = './patchnetvlad/configs/speed.ini'
    configfile = os.path.join(file_config.patchnetvladDir, 'configs/speed.ini')
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = True

    device = torch.device("cuda" if cuda else "cpu")

    encoder_dim, encoder = get_backend()

    # must resume to do extraction
    resume_ckpt = config['global_params']['resumePath'] + config['global_params']['num_pcs'] + '.pth.tar'

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] == int(config['global_params']['num_pcs'])
        config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

        model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=True)

        if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
            model.encoder = nn.DataParallel(model.encoder)
            # if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)

        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        print("=> loaded checkpoint '{}'".format(resume_ckpt, ))
    else:
        raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_ckpt))

    im_one = cv2.imread(im_path, -1)
    if im_one is None:
        raise FileNotFoundError(im_path + " does not exist")

    # match_two(model, device, config, im_one, feature_dir)
    res = retrieval(model, device, config, im_one, feature_dir, gt_dir)

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs
    print('Done')
    return res

def retrieval(model, device, config, im_one, feature_dir, gt_dir):
    df = pd.read_csv(gt_dir, sep=' ', header=None)

    pool_size = int(config['global_params']['num_pcs'])

    model.eval()
    print('check model device: ', next(model.parameters()).device)

    it = input_transform(
        (int(config['feature_extract']['imageresizeH']), int(config['feature_extract']['imageresizeW'])))

    im_one_pil = Image.fromarray(cv2.cvtColor(im_one, cv2.COLOR_BGR2RGB))
    im_one_pil = it(im_one_pil).unsqueeze(0)
    im_one_pil = im_one_pil.to(device)
    input_data = im_one_pil

    tqdm.write('====> Extracting Features')
    time1 = time.time()
    with torch.no_grad():
        image_encoding = model.encoder(input_data)
        temp = time.time()
        vlad_local, _ = model.pool(image_encoding)
        time2 = time.time()

        local_feats_one = []
        for this_iter, this_local in enumerate(vlad_local):
            this_local_feats = get_pca_encoding(model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))). \
                reshape(this_local.size(2), this_local.size(0), pool_size).permute(1, 2, 0)
            local_feats_one.append(torch.transpose(this_local_feats[0, :, :], 0, 1))

    tqdm.write('====> Calculating Keypoint Positions')
    patch_sizes = [int(s) for s in config['global_params']['patch_sizes'].split(",")]
    strides = [int(s) for s in config['global_params']['strides'].split(",")]
    patch_weights = np.array(config['feature_match']['patchWeights2Use'].split(",")).astype(float)

    all_keypoints = []
    all_indices = []

    tqdm.write('====> Matching Local Features')
    for patch_size, stride in zip(patch_sizes, strides):
        # we currently only provide support for square patches, but this can be easily modified for future works
        keypoints, indices = calc_keypoint_centers_from_patches(config['feature_match'], patch_size, patch_size, stride,
                                                                stride)
        all_keypoints.append(keypoints)
        all_indices.append(indices)

    matcher = PatchMatcher(config['feature_match']['matcher'], patch_sizes, strides, all_keypoints,
                           all_indices)
    # matching query with db feature
    # feature_dir = '/media/yushichen/LENOVO_USB_HDD/dataset/tum_sfm/output_feature'
    db_feature_list = os.listdir(feature_dir)
    db_feature_list.sort()
    place1 = time.time()
    # similar_table = {}
    max_similar, max_image_name = 0.0, ''
    for i in tqdm(range(0, len(db_feature_list), 10)):
        if db_feature_list[i] == 'globalfeats.npy':
            continue
        local_feats_two_np = np.load(os.path.join(feature_dir, db_feature_list[i]))
        local_feats_two = [torch.from_numpy(local_feats_two_np).to(device)]
        scores, inlier_keypoints_one, inlier_keypoints_two = matcher.match(local_feats_one, local_feats_two)
        score = -apply_patch_weights(scores, len(patch_sizes), patch_weights)
        file_name = 'images/f' + db_feature_list[i].lstrip('patchfeats_psize5_').rstrip('npy') + 'png'
        # similar_table[file_name] = score
        # print(f"Similarity score between the two images is: {score:.5f}. Larger scores indicate better matches.",
        #       i, file_name)
        if score > max_similar:
            max_similar = score
            max_image_index = i
    # print('max similar: ', max_similar, max_image_name)
    # print(df.loc[max_image_name])


    # print(sorted(similar_table))
    place2 = time.time()

    time3 = time.time()
    print('extracting time: ', time2 - time1)
    print('matching time: ', place2 - place1)
    return df.loc[max_image_index], max_similar


def relocalize_for_trial(image_data, feature_dir, gt_dir):
    print('Start relocalization!!!')
    configfile = './patchnetvlad/configs/speed.ini'
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = True

    device = torch.device("cuda" if cuda else "cpu")

    encoder_dim, encoder = get_backend()

    # must resume to do extraction
    resume_ckpt = config['global_params']['resumePath'] + config['global_params']['num_pcs'] + '.pth.tar'

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] == int(config['global_params']['num_pcs'])
        config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

        model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=True)

        if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
            model.encoder = nn.DataParallel(model.encoder)
            # if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)

        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        print("=> loaded checkpoint '{}'".format(resume_ckpt, ))
    else:
        raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_ckpt))

    # im_one = cv2.imread(im_path, -1)
    im_one = np.asarray(image_data)
    if im_one is None:
        raise FileNotFoundError("File does not exist")

    # match_two(model, device, config, im_one, feature_dir)
    res = retrieval(model, device, config, im_one, feature_dir, gt_dir)

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs
    print('Done')
    return res


if __name__ == "__main__":
    main()