from django.shortcuts import render
from .models import Developer, ApplicationFeature


def home(request):
    developers = Developer.objects.all()
    features = ApplicationFeature.objects.all()
    return render(request, 'home/home.html', {'developers': developers, 'features': features})


def search(request):
    if request.method == "POST":
        query = request.POST.get('query')
        # Logic for search goes here
        results = []  # Replace with actual search results
        return render(request, 'home/results.html', {'results': results})

    return render(request, 'home/search.html')

