# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
import os.path
import cv2
import torch
from torch import nn
from matching import Matching
from superpoint.superpoint import SuperPoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SuperpointNet(torch.nn.Module):
    def __init__(self,config, device):
        super(SuperpointNet, self).__init__()
        self.superpoint = SuperPoint(config.get('superpoint')).to(device)

    def forward(self,image):
        with torch.no_grad():
            pred = self.superpoint({'image':image})
        kp = pred['keypoints'][0].cpu().detach().numpy() # nx2 numpy
        desc = pred['descriptors'][0].cpu().detach().numpy().T
        scores = pred['scores'][0].cpu().detach().numpy()
        keypoint = []
        for i in range(len(kp)):
            keypoint.append(cv2.KeyPoint(kp[i][0], kp[i][1], scores[i]))
        keypoint = tuple(keypoint)
        return keypoint, desc


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def superpoint_test():
    path = '/media/yushichen/LENOVO_USB_HDD/projects/VisualOdometry/ipin_1/image_l/frame_000000.png'
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1,
        },
        'brute-force': {}
    }

    superpoint = SuperpointNet(config, device)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # frame2tensor
    frame = frame2tensor(image, device)
    res = superpoint(frame)
    print('Finish!')



if __name__ == '__main__':
    path_0 = '/media/yushichen/LENOVO_USB_HDD/projects/VisualOdometry/ipin_1/image_l/frame_000000.png'
    path_1 = '/media/yushichen/LENOVO_USB_HDD/projects/VisualOdometry/ipin_1/image_l/frame_000001.png'
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    image0 = cv2.imread(path_0, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE)
    frame0 = frame2tensor(image0, device)
    frame1 = frame2tensor(image1, device)
    matcher = Matching(config).eval().to(device)
    pred = matcher({'image0': frame1, 'image1': frame0})
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().detach().numpy()
    i = 0