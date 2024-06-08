import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

import math
import os
import numpy as np
import sys
import json
import csv
from typing import List, Tuple

def load_features_from_json(input_file: str) -> List[Tuple[str, np.ndarray]]:
    """
    Load features from a JSON file.

    Parameters:
    input_file (str): Path to the input JSON file.

    Returns:
    List[Tuple[str, np.ndarray]]: A list where each element is a tuple containing the file path and the features as a numpy array.
    """
    with open(input_file, "r") as file:
        data = json.load(file)
        features = [(image[0], np.array(image[1])) for image in data]
    return features

def euclidean_distance(l1: np.ndarray, l2: np.ndarray) -> float:
    return np.linalg.norm(l1 - l2)

def chi_square_distance(l1: np.ndarray, l2: np.ndarray) -> float:
    
    epsilon = 1e-10
    l2_safe = np.where(l2 == 0, epsilon, l2)
    
    return np.sum((l1 - l2)**2 / l2_safe)

def bhattacharyya_distance(l1: np.ndarray, l2: np.ndarray) -> float:
    
    N_1 = np.mean(l1)
    N_2 = np.mean(l2)

    score = np.sum(np.sqrt(l1 * l2))

    den = np.sqrt(N_1 * N_2 * len(l1) * len(l2))

    distance = np.sqrt(1 - (score / den))

    return distance

def get_k_nearest_neighbors(features: List[Tuple[str, np.ndarray]], test: Tuple[str, np.ndarray], k: int, function= int) -> List[Tuple[str, np.ndarray, float]]:
    if function == 0:
        distances = [(name, feat, euclidean_distance(test[1], feat)) for name, feat in features]
    elif function == 1:
        distances = [(name, feat, chi_square_distance(test[1], feat)) for name, feat in features]
    elif function == 2:
        distances = [(name, feat, bhattacharyya_distance(test[1], feat)) for name, feat in features]
    distances.sort(key=lambda x: x[2])
    return distances[:k]

def search_with_filename(filename: str, top: int = 20, model = "VGG16") -> Tuple[str, List[str], List[str]]:
    # The model could be VGG16, ResNet50 or MobileNet
    features = load_features_from_json(f'Features_train/{model}.json')
    image_req = next(i for i, (path, _) in enumerate(features) if os.path.basename(path) == filename)
    return search(image_req, top, model)

def search(image_req: int, top: int = 20, model = "VGG16") -> Tuple[str, List[str], List[str]]:
    features = load_features_from_json(f'Features_train/{model}.json')
    # distance_function = 0 for euclidean, 1 for chi-square, 2 for bhattacharyya
    distance_function = 0 
    neighbors = get_k_nearest_neighbors(features, features[image_req], top, distance_function)
    
    query_image_path = features[image_req][0]
    query_image_name = os.path.splitext(os.path.basename(query_image_path))[0]
    
    close_image_paths = [neighbor[0] for neighbor in neighbors]
    close_image_names = [os.path.splitext(os.path.basename(path))[0] for path in close_image_paths]

    # Display query image
    plt.figure(figsize=(5, 5))
    plt.imshow(imread(query_image_path), cmap='gray', interpolation='none')
    plt.title("Query Image")
    plt.show()

    # Display nearest neighbor images
    plt.figure(figsize=(25, 25))
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    for j, path in enumerate(close_image_paths):
        plt.subplot(int(top/4), int(top/5), j + 1)
        plt.imshow(imread(path), cmap='gray', interpolation='none')
        plt.title(f"Close Image {j+1}")
    plt.show()

    return query_image_name, close_image_paths, close_image_names

def compute_rp(file_path: str, top: int, query_image_name: str, close_image_names: List[str]):
    position1 = int(query_image_name) // 100
    rp = []

    for i in range(top):
        relevant_count = sum(1 for j in range(i + 1) if int(close_image_names[j]) // 100 == position1)
        precision = (relevant_count / (i + 1)) * 100
        recall = (relevant_count / top) * 100
        rp.append(f"{precision} {recall}")

    with open(file_path, 'w') as f:
        f.write("\n".join(rp))

def display_rp(file_path: str, model: str):
    x, y = [], []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    plt.figure()
    plt.plot(y, x, 'C1', label=f"{model}")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Recall/Precision Curve")
    plt.legend()
    plt.show()

def main(image_req: int):
    if not (0 <= image_req <= 1000):
        raise ValueError("The number should be between 0 and 1000")
    models = ["VGG16", "ResNet50", "MobileNet"]
    model = models[2]
    query_image_name, close_image_paths, close_image_names = search_with_filename(f"{image_req}.jpg", 20, model)

    rp_file_path = f"{model}_RP.txt"
    compute_rp(rp_file_path, 20, query_image_name, close_image_names)
    display_rp(rp_file_path, model)

if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print("Usage: python script.py <image_req>")
        sys.exit(1)

    image_req = int(sys.argv[1])
    main(image_req)
