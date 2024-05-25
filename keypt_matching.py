import cv2
import os
import numpy as np 
import json

DATA_PATH = "./preprocessed-data/images"
OUTPUT_FILE = "point_correspondences.json"

def load_images(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    return images

def filter_matches(matches, thresh=0.70):
    filtered = []
    for m in matches:
        if m[0].distance / m[1].distance < thresh:
            filtered.append(m[0])
    return filtered

def main():
    images = load_images(DATA_PATH)
    sift = cv2.SIFT_create()
    
    keypoints_list = []
    descriptors_list = []
    
    for i, image in enumerate(images):
        keypoints, descriptors = sift.detectAndCompute(image, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
        print(f"SIFT frame {i}")
    
    print("Features detection finished.")
    
    point_correspondences = []
    for i in range(0, len(images)-1):
        matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(
            descriptors_list[i].astype(np.float32), 
            descriptors_list[i+1].astype(np.float32), 
            k=2
        )
        filtered_matches = filter_matches(matches)
        
        correspondences = []
        for match in filtered_matches:
            idx1 = match.queryIdx
            idx2 = match.trainIdx
            p1 = keypoints_list[i][idx1].pt
            p2 = keypoints_list[i+1][idx2].pt
            
            correspondences.append([[float(p1[0]), float(p1[1])], 
                                    [float(p2[0]), float(p2[1])]])
        
        point_correspondences.append(correspondences)
        print(f"Correspondences frame {i}")
        print(f"# pts: {len(correspondences)}")

    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(point_correspondences, f)
    
    print("Corespondence detection finished.")


if __name__ == "__main__":
    main()
