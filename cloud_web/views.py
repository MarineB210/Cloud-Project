from django.shortcuts import render, redirect
from django.urls import reverse
from appli import main

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
    query_image = request.GET.get('query-image', '')
    top = request.GET.get('top', '')
    descriptors = request.GET.get('descriptors', '').split(',')
    print(f'Query image: {query_image}')  # Debug print statement
    print(f'Top: {top}')  # Debug print statement
    print(f'Descriptors: {descriptors}')  # Debug print statement

    if query_image and top and descriptors:
        # Process the data here
        results = main()
        return render(request, 'results.html', {'query_image': query_image, 'top': top, 'descriptors': descriptors, 'results': results})
    else:
        return render(request, 'results.html', {'error': 'No query provided.'})
