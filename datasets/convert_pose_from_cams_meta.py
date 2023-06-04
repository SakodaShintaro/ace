""" A script to convert pose npy to txt
"""

import argparse
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pose_npy', type=str)
    parser.add_argument('output_dir', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    npy_data = np.load(args.pose_npy)
    poses = npy_data[:, 0:12]
    poses = poses.reshape(-1, 3, 4)
    k = npy_data[:, 12:21]
    k = k.reshape(-1, 3, 3)

    # add 4th row to poses
    row = np.zeros((poses.shape[0], 1, 4))
    row[:, :, 3] = 1
    poses = np.concatenate([poses, row], axis=1)
    pose_0 = poses[0].copy()
    pose_0[0:3, 0:3] = np.zeros((3, 3))
    pose_0[3][3] = 0

    output_dir_pose = f"{args.output_dir}/poses"
    os.makedirs(output_dir_pose, exist_ok=True)
    output_dir_calib = f"{args.output_dir}/calibration"
    os.makedirs(output_dir_calib, exist_ok=True)

    for i, pose in enumerate(poses):
        np.savetxt(f"{output_dir_pose}/{i:08d}.txt", pose - pose_0, fmt='%.6f')
        curr_k = k[i]
        focal = (curr_k[0, 0] + curr_k[1, 1]) / 2
        focal = np.array([focal])
        np.savetxt(f"{output_dir_calib}/{i:08d}.txt", focal, fmt='%.6f')
