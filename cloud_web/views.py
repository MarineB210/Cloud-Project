import os
import numpy as np
import json
import csv
from django.conf import settings
import matplotlib.pyplot as plt

from django.shortcuts import render, redirect
from django.urls import reverse

from matplotlib.pyplot import imread

from typing import List, Tuple


def home(request):
    return render(request, 'home.html', {})


def search(request):
    if request.method == "POST":
        query_image = request.POST.get('query-image', '')
        top = request.POST.get('top', '')
        descriptors = request.POST.getlist('descriptor')
        if query_image and top and descriptors:
            descriptors_str = ','.join(descriptors)
            return redirect(f"{reverse('results')}?query-image={query_image}&top={top}&descriptors={descriptors_str}")
        else:
            return render(request, 'search.html', {'error': 'Please fill in all fields.'})
    return render(request, 'search.html')


def results(request):
    query_image = int(request.GET.get('query-image', '').split('.')[0])
    top = int(request.GET.get('top', ''))
    descriptors = request.GET.get('descriptors', '').split(',')
    print(f'Query image: {query_image}')  # Debug print statement
    print(f'Top: {top}')  # Debug print statement
    print(f'Descriptors: {descriptors}')  # Debug print statement

    if query_image and top and descriptors:
        # Process the data here
        query_image_name, close_image_paths, close_image_names = search_images(query_image, top, descriptors[0])
        results_images = []
        for path in close_image_paths:
            results_images.append(os.path.join(settings.MEDIA_URL, path))

        for descriptor in descriptors:
            compute_rp(descriptor+'.txt', top, query_image_name, close_image_names)
            display_rp(descriptor+'.txt')

        return render(request, 'results.html', {'query_image': query_image, 'top': top, 'descriptors': descriptors, 'results_images': results_images})
    else:
        return render(request, 'results.html', {'error': 'No query provided.'})


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


def get_k_nearest_neighbors(features: List[Tuple[str, np.ndarray]], test: Tuple[str, np.ndarray], k: int) -> List[
    Tuple[str, np.ndarray, float]]:
    distances = [(name, feat, euclidean_distance(test[1], feat)) for name, feat in features]
    distances.sort(key=lambda x: x[2])
    return distances[:k]


def search_images(image_req: int, top: int, descriptor: str) -> Tuple[str, List[str], List[str]]:
    features = load_features_from_json(f"Features_train/{descriptor}.json")
    neighbors = get_k_nearest_neighbors(features, features[image_req], top)

    query_image_path = features[image_req][0]
    query_image_name = os.path.splitext(os.path.basename(query_image_path))[0]

    close_image_paths = [neighbor[0] for neighbor in neighbors]
    close_image_names = [os.path.splitext(os.path.basename(path))[0] for path in close_image_paths]

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


def display_rp(file_path: str):
    x, y = [], []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    plt.figure()
    plt.plot(y, x, 'C1', label="VGG16")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Recall/Precision Curve")
    plt.legend()
    plt.savefig('media/results_plot.png')
