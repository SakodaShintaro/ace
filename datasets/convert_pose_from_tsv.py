import argparse
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import os
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_pose_tsv", type=str)
    parser.add_argument('output_dir', type=str)
    return parser.parse_args()


def load_camera_info_from_yaml(filename):
    with open(filename, "r") as input_file:
        camera_info_dict = yaml.safe_load(input_file)
        camera_info_dict["D"] = np.array(camera_info_dict["D"])
        camera_info_dict["K"] = np.array(camera_info_dict["K"]).reshape((3, 3))
        camera_info_dict["R"] = np.array(camera_info_dict["R"]).reshape((3, 3))
        camera_info_dict["P"] = np.array(camera_info_dict["P"]).reshape((3, 4))
        return camera_info_dict


AXIS_CONVERT_MAT_A2B = np.array(
    [[0, 0, +1, 0],
     [-1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, 0, 1]], dtype=np.float64
)

if __name__ == "__main__":
    args = parse_args()

    path_to_pose_tsv = args.path_to_pose_tsv
    target_dir = os.path.dirname(path_to_pose_tsv)

    df_pose = pd.read_csv(path_to_pose_tsv, sep="\t", index_col=0)
    n = len(df_pose)
    scale = 100
    pose_xyz = df_pose[['x', 'y', 'z']].values / scale
    pose_quat = df_pose[['qx', 'qy', 'qz', 'qw']].values
    rotation_mat = Rotation.from_quat(pose_quat).as_matrix()
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0:3, 0:3] = rotation_mat
    poses[:, 0:3, 3:4] = pose_xyz.reshape((n, 3, 1))

    # convert axis
    poses = AXIS_CONVERT_MAT_A2B.T @ poses @ AXIS_CONVERT_MAT_A2B

    # save camera meta
    camera_info = load_camera_info_from_yaml(f"{target_dir}/camera_info.yaml")
    k = camera_info["K"]
    k = np.tile(k, (n, 1, 1))
    k = k.reshape((n, 3, 3))

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
