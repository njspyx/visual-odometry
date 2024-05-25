import numpy as np

def normalize(x, y):
    """
    Function: find the transformation T to make coordinates zero mean and the variance as sqrt(2)
    Input: x, y - coordinates
    Output: normalized coordinates, transformation T
    """
    # YOUR CODE HERE:
    # Zero-mean the coordinates
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_norm = x - x_mean
    y_norm = y - y_mean

    # Compute the average distance to center
    d = np.mean(np.sqrt(x_norm**2 + y_norm**2))

    # Scale the coordinates using distance
    s = np.sqrt(2) / d
    x_norm = x_norm * s
    y_norm = y_norm * s

    # Construct T
    T = np.array([[s, 0, -x_mean * s], [0, s, -y_mean * s], [0, 0, 1]])

    return np.concatenate((x_norm, y_norm), axis=1), T


def ransacF(x1, y1, x2, y2, num_points=8, num_iterations=1000, threshold=0.01):
    """
    Find normalization matrix
    Transform point set 1 and 2
    RANSAC based 8-point algorithm
    Input:
        x1, y1, x2, y2 - coordinates
        num_iterations - how many iterations
        threshold - threshold for inlier check
    Output:
        Best fundamental matrix
        corresponding inlier indices
    """
    # YOUR CODE HERE:

    max_inliers = 0
    best_F = None
    best_inliers = None

    # Hint:
    # for ... in num_iterations:
    for i in range(num_iterations):
    #    1. Randomly select 8 points
        indices = np.random.choice(x1.shape[0], num_points, replace=False)
        select_x1 = x1[indices]
        select_y1 = y1[indices]
        select_x2 = x2[indices]
        select_y2 = y2[indices]

    #    2. Call computeF()
        F = computeF(select_x1, select_y1, select_x2, select_y2)

    #    3. Call getInliers()
        curr_inliers = getInliers(x1, y1, x2, y2, F, threshold)

    #    4. Update F and inliers.
        if len(curr_inliers) > max_inliers:
            max_inliers = len(curr_inliers)
            best_F = F
            best_inliers = curr_inliers

    return best_F, best_inliers


def computeF(x1, y1, x2, y2):
    """
    Function: compute fundamental matrix from corresponding points
    Input:
        x1, y1, x2, y2 - coordinates
    Output:
        fundamental matrix, 3x3
    """
    # Make matrix A
    n = x1.shape[0]
    A = np.zeros((n, 9))

    A[:, 0] = x1 * x2
    A[:, 1] = x1 * y2
    A[:, 2] = x1
    A[:, 3] = y1 * x2
    A[:, 4] = y1 * y2
    A[:, 5] = y1
    A[:, 6] = x2
    A[:, 7] = y2
    A[:, 8] = 1

    # 2. Do SVD for A
    U, S, VT = np.linalg.svd(A)
    # 3. Find fundamental matrix F
    f = VT.T[:, -1]
    F = f.reshape(3, 3).T

    # 4. Resolve det(F)=0
    U, S, VT = np.linalg.svd(F)
    new_S = np.diag([S[0], S[1], 0])
    F = U @ new_S @ VT

    return F


def getInliers(x1, y1, x2, y2, F, thresh):
    """
    Function: implement the criteria checking inliers.
    Input:
        x1, y1, x2, y2 - coordinates
        F - estimated fundamental matrix, 3x3
        thresh - threshold for passing the error
    Output:
        inlier indices
    """
    # YOUR CODE HERE:
    x1 = x1.reshape(-1, 1)
    y1 = y1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    y2 = y2.reshape(-1, 1)
    points1 = np.hstack((x1, y1, np.ones(x1.shape)))
    points2 = np.hstack((x2, y2, np.ones(x2.shape)))

    epipolar_lines = np.dot(F, points1.T).T

    distances = np.abs(np.sum(epipolar_lines * points2, axis=1)) / np.sqrt(epipolar_lines[:, 0]**2 + epipolar_lines[:, 1]**2)

    inlier_indices = np.where(distances < thresh)[0]

    return inlier_indices

def findFundamentalMatCustom(pts1, pts2):
    x1, y1 = pts1[:, 0:1], pts1[:, 1:2]
    x2, y2 = pts2[:, 0:1], pts2[:, 1:2]
    norm_p1, T1 = normalize(x1, y1)
    norm_x1, norm_y1 = norm_p1[:, :1].flatten(), norm_p1[:, 1:2].flatten()
    norm_p2, T2 = normalize(x2, y2)
    norm_x2, norm_y2 = norm_p2[:, :1].flatten(), norm_p2[:, 1:2].flatten()

    p = 0.99
    e = 0.5
    s = 8
    num_iter = int(np.log(1 - p) / np.log(1 - (1 - e)**s))
    F_norm, _ = ransacF(norm_x1, norm_y1, norm_x2, norm_y2, num_iterations=num_iter, threshold=0.1)

    F = T2.T @ F_norm @ T1
    F = F / F[2, 2]

    return F

    