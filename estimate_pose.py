import numpy as np
import cv2

def extract_poses(E):
    """
    Function: extract the rotation and translation from the essential matrix.
    Input: E - essential matrix, 3x3
    Output: All possible rotation and translation pairs.
    """
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], 
                  [1, 0, 0], 
                  [0, 0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W @ Vt
    R3 = U @ W.T @ Vt
    R4 = U @ W.T @ Vt

    C1 = U[:, 2]
    C2 = -U[:, 2]
    C3 = U[:, 2]
    C4 = -U[:, 2]

    poses = [(C1, R1), (C2, R2), (C3, R3), (C4, R4)]
    return poses

def linear_triangulation(P1, P2, pts1, pts2):
    """
    Function: linear triangulation method.
    Input:
        P1, P2 - projection matrices, 3x4
        pts1, pts2 - corresponding points, Nx2
    Output: 3D points, Nx3
    """
    N = pts1.shape[0]
    pts3D = np.zeros((N, 3))

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :4]
        X = X / X[3]
        pts3D[i] = X[:3]

    return pts3D

def check_cheirality(C, R, X):
    """
    Function: check the cheirality condition for a given pose.
    Input:
        C - camera center, 3x1
        R - rotation matrix, 3x3
        pts3D - 3D points, Nx3
    Output: number of points satisfying the cheirality condition
    """
    N = X.shape[0]
    count = 0

    for i in range(N):
        X_i = X[i]
        if np.dot(R[2], (X_i - C) / np.linalg.norm(X_i - C)) > 0 and X_i[2] > 0:
            count += 1

    return count

def recoverPoseCustom(E, pts1, pts2, K):
    """
    Function: recover the pose from the essential matrix.
    Input:
        E - essential matrix, 3x3
        pts1, pts2 - corresponding points, Nx2
        K - intrinsic camera matrix, 3x3
    Output:
        Best rotation and translation
        R - rotation matrix, 3x3
        C - camera center, 3x1
    """
    poses = extract_poses(E)
    best_count = 0
    best_pose = None

    for C, R in poses:
        if np.linalg.det(R) < 0:
            C = -C
            R = -R

        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, -R @ C.reshape(3, 1)))
        X = linear_triangulation(P1, P2, pts1, pts2)
        count = check_cheirality(C, R, X)

        if count > best_count:
            best_count = count
            best_pose = (C, R)

    if best_pose is None:
        raise ValueError("No pose found satisfying the cheirality condition.")

    C, R = best_pose
    T = -R @ C.reshape(3, 1)   

    return R, T