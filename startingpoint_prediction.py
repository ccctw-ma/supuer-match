import os

import config
from match_query import relocalize, relocalize_for_trial
import cv2

# db_dir = '/media/yushichen/DATA/IPIN_res/Site_B'
# db_dir = '/media/yushichen/LENOVO_USB_HDD/IPIN_res/site_A'
db_dir = config.siteDir


# test_image_path = '/media/yushichen/LENOVO_USB_HDD/IPIN_res/site_A/seq_1/query/frame_000962.png'
# images/frame_000961.png 127.370288 36.383524 3 0.054208 -0.466538 0.881656 -0.045667 SM-N986N
def starting_point_prediction(image_path):
    # im_one = cv2.imread('/media/yushichen/LENOVO_USB_HDD/IPIN_res/site_A/seq_1/query/frame_000962.png', -1)
    siteA = os.listdir(db_dir)
    max_similar = 0
    max_info = []
    # for seq in siteA:
    #     feature_dir = os.path.join(db_dir, seq, 'output_feature')
    #     gt_dir = os.path.join(db_dir, seq, 'pose_unit.txt')
    #     res = relocalize(image_path, feature_dir, gt_dir)
    #     if res[-1] > max_similar:
    #         max_similar = res[-1]
    #         max_info = [res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5],
    #                     res[0][6], res[0][7], res[0][8], res[0][9], res[0][10], res[0][11],
    #                     res[0][12], res[0][13], res[0][14]]
    feature_dir = os.path.join(db_dir, 'output_feature')
    gt_dir = os.path.join(db_dir, 'pose_unit.txt')
    res = relocalize(image_path, feature_dir, gt_dir)
    if res[-1] > max_similar:
        max_similar = res[-1]
        max_info = [res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5],
                    res[0][6], res[0][7], res[0][8], res[0][9], res[0][10], res[0][11],
                    res[0][12], res[0][13], res[0][14]]

    print(max_similar)
    print(max_info)
    return max_info


def starting_point_prediction_for_trial(image_data):
    # im_one = cv2.imread('/media/yushichen/LENOVO_USB_HDD/IPIN_res/site_A/seq_1/query/frame_000962.png', -1)
    siteA = os.listdir(db_dir)
    max_similar = 0
    max_info = []
    for seq in siteA:
        feature_dir = os.path.join(db_dir, seq, 'output_feature')
        gt_dir = os.path.join(db_dir, seq, 'pose_unit.txt')
        res = relocalize_for_trial(image_data, feature_dir, gt_dir)
        if res[-1] > max_similar:
            max_similar = res[-1]
            max_info = [res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5],
                        res[0][6], res[0][7], res[0][8], res[0][9], res[0][10], res[0][11],
                        res[0][12], res[0][13], res[0][14]]
    print(max_similar)
    print(max_info)
    return max_info, max_similar


if __name__ == '__main__':
    imagePath = os.path.join(config.dataDir, 'images/frame_001160.png')
    starting_point_prediction(imagePath)
