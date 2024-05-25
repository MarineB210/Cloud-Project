from django.shortcuts import render
from .models import Developer, ApplicationFeature

def home(request):
    #developers = Developer.objects.all()
    #features = ApplicationFeature.objects.all()
    return render(request, 'home.html', {})

def search(request):
    if request.method == "POST":
        query = request.POST.get('query')
        # Logic for search goes here
        results = []  # Replace with actual search results
        return render(request, 'results.html', {'results': results})

    return render(request, 'search.html')

