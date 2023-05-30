from django.shortcuts import render
from django.http import JsonResponse
from .utils import calculate_gpt

def home(request):
    return render(request, 'home.html')

def calculate_ai_content(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        ai_percentage = calculate_gpt(text)
        return JsonResponse({'ai_percentage': ai_percentage})
    return JsonResponse({'error': 'Invalid request method'})
