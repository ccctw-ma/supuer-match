import os
import numpy as np
from tqdm import tqdm

import angel_trans
import pymap3d as pm

import config

# 序列1第一帧经纬度
b_lon = 127.370282
b_lat = 36.383650

floor = np.array([3]).reshape(1, 1)
path = config.dataDir
f = open(os.path.join(path, "dataset_all.txt"), "r")
msgs = f.readlines()
poses = []
coordinate = []
for msg in tqdm(msgs, unit="lines"):
    m = msg.split(" ")
    print(m)
    lon = float(m[1])
    lat = float(m[2])
    x, y, z = pm.geodetic2enu(lat, lon, 0, b_lat, b_lon, 0, deg=True)
    coordinate.append((x, y, z))
    q = [m[4], m[5], m[6], m[7]]
    euler = angel_trans.quaternion2euler(q)
    rot = angel_trans.euler2rotation(euler)
    trans = np.array([x, y, 0]).reshape(3, 1)
    print(euler, rot)
    pose = np.concatenate((rot, trans), axis=1).reshape(1, 12)
    # 加上经纬度
    pose = np.concatenate((np.array([lon, lat]).reshape(1, 2), pose), axis=1)
    pose = np.append(pose, floor, axis=1)
    poses.append(pose)
    # 数据格式 pose[0:2] latlon  pose[2:11] rotation pose[11:14] transition pose[14] floor

poses = np.concatenate(poses, axis=0)
np.savetxt(os.path.join(path, "pose_unit.txt"), poses, delimiter=' ', fmt='%1.8e')
for c in coordinate:
    print(c)