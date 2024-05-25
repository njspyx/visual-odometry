import os
import numpy as np
import cv2
from scipy.ndimage import map_coordinates as interp2

DATA_PATH = "./Oxford_dataset_reduced/images"
CAMERA_MODEL_PATH = "./Oxford_dataset_reduced/model"
OUTPUT_PATH = "./preprocessed-data"

def ReadCameraModel(models_dir):
    """
    Author: Kanishka Ganguly
    """
    intrinsics_path = models_dir + "/stereo_narrow_left.txt"
    lut_path = models_dir + "/stereo_narrow_left_distortion_lut.bin"

    intrinsics = np.loadtxt(intrinsics_path)
    # Intrinsics
    fx = intrinsics[0,0]
    fy = intrinsics[0,1]
    cx = intrinsics[0,2]
    cy = intrinsics[0,3]
    # 4x4 matrix that transforms x-forward coordinate frame at camera origin and image frame for specific lens
    G_camera_image = intrinsics[1:5,0:4]
    # LUT for undistortion
    # LUT consists of (u,v) pair for each pixel)
    lut = np.fromfile(lut_path, np.double)
    lut = lut.reshape([2, lut.size//2])
    LUT = lut.transpose()

    return fx, fy, cx, cy, G_camera_image, LUT

def UndistortImage(image, LUT):
    """
    Author: Kanishka Ganguly
    """
    reshaped_lut = LUT[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
    undistorted = np.rollaxis(np.array([interp2(image[:, :, channel], reshaped_lut, order=1)
                                for channel in range(0, image.shape[2])]), 0, 3)

    
    return undistorted.astype(image.dtype)

# Compute Instrinsic Matrix
fx, fy, cx, cy, _, LUT = ReadCameraModel(CAMERA_MODEL_PATH)
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
])
model_out_path = os.path.join(OUTPUT_PATH, "model")
os.makedirs(model_out_path, exist_ok=True)
np.save(os.path.join(model_out_path, "LUT.npy"), LUT)
np.save(os.path.join(model_out_path, "K.npy"), K)


# Load and Demosaic Images
file_list = sorted([file for file in os.listdir(DATA_PATH) if file.endswith(".png")])
data_out_path = os.path.join(OUTPUT_PATH, "images")
os.makedirs(data_out_path, exist_ok=True)

for i, filename in enumerate(file_list):
    file_path = os.path.join(DATA_PATH, filename)
    image = cv2.imread(file_path, flags=-1)
    
    processed_image = cv2.cvtColor(image, cv2.COLOR_BayerGR2BGR)
    processed_image = UndistortImage(processed_image, LUT)[:800, :]
    cv2.imwrite(os.path.join(data_out_path, f"{i:03d}.png"), processed_image)
    # print(f"{i:03d}.png")

print("Preprocessing complete.")
