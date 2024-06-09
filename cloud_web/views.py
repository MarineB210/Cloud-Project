import os
import numpy as np
import json
import csv
from django.conf import settings
import matplotlib.pyplot as plt

from django.shortcuts import render, redirect
from django.urls import reverse
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required

from .forms import Loginform, RegisterForm
from typing import List, Tuple


@login_required
def home(request):
    return render(request, 'home.html', {})


def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = RegisterForm()
    return render(request, 'accounts/register.html', {'form': form})


def user_login(request):
    if request.method == 'POST':
        form = Loginform(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
    else:
        form = Loginform()
    return render(request, 'accounts/login.html', {'form': form})


@login_required
def search(request):
    if request.method == "POST":
        query_image = request.POST.get('query-image', '')
        top = request.POST.get('top', '')
        descriptors = request.POST.getlist('descriptor')
        distance = request.POST.get('metric', '')
        if query_image and top and descriptors and distance:
            descriptors_str = ','.join(descriptors)
            return redirect(f"{reverse('results')}?query-image={query_image}&top={top}&descriptors={descriptors_str}&distance={distance}")
        else:
            return render(request, 'search.html', {'error': 'Please fill in all fields.'})
    return render(request, 'search.html')


@login_required
def results(request):
    query_image = request.GET.get('query-image', '')
    top = int(request.GET.get('top', ''))
    descriptors = request.GET.get('descriptors', '').split(',')
    distance = int(request.GET.get('distance', ''))
    print(f'Query image: {query_image}')  # Debug print statement
    print(f'Top: {top}')  # Debug print statement
    print(f'Descriptors: {descriptors}')  # Debug print statement

    if query_image and top and descriptors:
        # Process the data here
        query_image_name, close_image_paths, close_image_names = search_with_filename(query_image, top, descriptors[0], distance)
        results_images = []
        for path in close_image_paths:
            results_images.append(os.path.join(settings.MEDIA_URL, path))

        compute_rp(descriptors[0]+'.txt', top, query_image_name, close_image_names)
        display_rp(descriptors[0]+'.txt', descriptors[0])

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


def chi_square_distance(l1: np.ndarray, l2: np.ndarray) -> float:
    epsilon = 1e-10
    l2_safe = np.where(l2 == 0, epsilon, l2)

    return np.sum((l1 - l2) ** 2 / l2_safe)


def bhattacharyya_distance(l1: np.ndarray, l2: np.ndarray) -> float:
    N_1 = np.mean(l1)
    N_2 = np.mean(l2)

    score = np.sum(np.sqrt(l1 * l2))

    den = np.sqrt(N_1 * N_2 * len(l1) * len(l2))

    distance = np.sqrt(1 - (score / den))

    return distance

def get_k_nearest_neighbors(features: List[Tuple[str, np.ndarray]], test: Tuple[str, np.ndarray], k: int, distance_function: int) -> List[Tuple[str, np.ndarray, float]]:
    distances = []
    if distance_function == 0:
        distances = [(name, feat, euclidean_distance(test[1], feat)) for name, feat in features]
    elif distance_function == 1:
        distances = [(name, feat, chi_square_distance(test[1], feat)) for name, feat in features]
    elif distance_function == 2:
        distances = [(name, feat, bhattacharyya_distance(test[1], feat)) for name, feat in features]
    distances.sort(key=lambda x: x[2])
    return distances[:k]

def search_with_filename(filename: str, top: int, descriptor: str, distance_function: int) -> Tuple[str, List[str], List[str]]:
    features = load_features_from_json(f"Features_train/{descriptor}.json")
    image_req = next(i for i, (path, _) in enumerate(features) if os.path.basename(path) == filename)
    return search_images(image_req, top, descriptor, distance_function)

def search_images(image_req: int, top: int, descriptor: str, distance_function: int) -> Tuple[str, List[str], List[str]]:
    features = load_features_from_json(f"Features_train/{descriptor}.json")
    neighbors = get_k_nearest_neighbors(features, features[image_req], top, distance_function)

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


def display_rp(file_path: str, descriptor: str):
    x, y = [], []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    plt.figure()
    plt.plot(y, x, 'C1', label=descriptor)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Recall/Precision Curve")
    plt.legend()
    plt.savefig('media/results_plot.png')
