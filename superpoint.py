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


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 500,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        # input channel is 1. Use grayscale image instead of RGB image
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        # path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        # path = 'C:/Users/Lenovo/msc/visual-position/ourVO/model_weight/superpoint_v1.pth'
        path = os.path.join(config.modelDir, 'superpoint_v1.pth')
        model = torch.load(str(path))
        self.load_state_dict(model)

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h * 8, w * 8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        # print(np.array(keypoints).shape) only has one dimension
        # print(keypoints[0].size())
        # print(keypoints[0].size(),descriptors[0].size(),scores[0].size())
        # print(scores)
        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }


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