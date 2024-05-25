import cv2
import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from estimate_fundamental import findFundamentalMatCustom
from estimate_pose_nonlin import recoverPoseCustom

DATA_PATH = "./preprocessed-data/images"
KEYPT_PATH = "point_correspondences.json"
CAMERA_MODEL_PATH = "./preprocessed-data/model/K.npy"

def load_correspondences(path):
    with open(path, "r") as f:
        return json.load(f)

def estimate_F(pts1, pts2):
    F, _ = cv2.findFundamentalMat(pts1,
                                  pts2,
                                  cv2.FM_RANSAC,
                                  ransacReprojThreshold=0.1,
                                  confidence=0.99)
    return F

def estimate_F_custom(pts1, pts2):
    return findFundamentalMatCustom(pts1, pts2)

def estimate_E(F, K):
    E = K.T @ F @ K
    U, _, VT = np.linalg.svd(E)
    S = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])
    E = U @ S @ VT
    return E
    
def main(use_custom=False):
    point_correpondences = load_correspondences(KEYPT_PATH)
    K = np.load(CAMERA_MODEL_PATH)
    R_list = []
    T_list = []
    
    print("Fundamental and essential matrix estimation:")
    for i in range(len(point_correpondences)):
        print(f"Frame {i}")
        pts1 = np.array(point_correpondences[i])[:, 0]
        pts2 = np.array(point_correpondences[i])[:, 1]
        
        if use_custom:
            F = estimate_F_custom(pts1, pts2)
        else:
            F = estimate_F(pts1, pts2)
        
        E = estimate_E(F, K)
        
        if use_custom:
            R, T = recoverPoseCustom(E, pts1, pts2, K)
        else:
            _, R, T, _ = cv2.recoverPose(E, pts1, pts2, K)
        R_list.append(R)
        T_list.append(T)
    
    # reconstruct trajectory
    print("Reconstructing trajectory:")
    trajectory = [(0, 0, 0)]
    # current_pos = np.array([[0, 0, 0, 1]]).T
    camera_pose = np.eye(4)
    
    for i in range(len(R_list)):
        print(f"Frame {i}")
        R = R_list[i]
        T = T_list[i]
        
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = T.flatten()
        
        camera_pose = transform @ camera_pose 
        # x, y, z = camera_pose[:3, -1]
        point = np.linalg.inv(camera_pose) @ np.array([0, 0, 0, 1]).T
        x, y, z = point[:3]
        trajectory.append((x, y, z))

    trajectory = np.array(trajectory)

    # Plot 3d
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='o', linestyle='-', color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if use_custom:
        ax.set_title('Camera Trajectory (Custom)')
        plt.savefig('camera_trajectory_3d_custom.png')
    else:
        ax.set_title('Camera Trajectory (OpenCV)')
        plt.savefig('camera_trajectory_3d.png')
    plt.show()

    # Plot 2d
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 2], marker='o', linestyle='-', color='blue')
    plt.xlabel('X')
    plt.ylabel('Z')
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate fundamental matrix and reconstruct trajectory')
    parser.add_argument('--custom', action='store_true', help='Use custom fundamental matrix estimator')
    args = parser.parse_args()
    main(args.custom)

