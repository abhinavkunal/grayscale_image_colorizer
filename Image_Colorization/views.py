from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from PIL import Image
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# Create your views here.
def solve(filename):
    gen = load_model('final_gen.hdf5')
    img_size = 120
    x, y = [], []
    path = "./media"
    path = os.path.join(path, filename)
    rgb_image = Image.open(path).resize((img_size, img_size)).convert('RGB')
    # Normalize the RGB image array
    rgb_img_array = (np.asarray(rgb_image)) / 255
    gray_image = rgb_image.convert('L')
    # Normalize the grayscale image array
    gray_img_array = (np.asarray(gray_image).reshape((img_size, img_size, 1))) / 255
    # Append both the image arrays
    x.append(gray_img_array)
    y.append(rgb_img_array)
    # x.append( gray_img_array )
    # y.append( rgb_img_array )
    x1, y1 = np.array(x), np.array(y)
    op = gen(x1[0:1]).numpy()
    i = 0
    image = Image.fromarray((op[i] * 255).astype('uint8'))#.resize((1024, 1024))
    image = np.asarray(image)
    fig = plt.imshow(image)
    plt.axis('off')
    plt.savefig('./media/converted/coloured.jpg', bbox_inches='tight', pad_inches=0)

def home(request):
    return render(request, 'index.html')

def upload(request):
    image = request.FILES['grayscale_image']
    print(image)
    fs = FileSystemStorage()
    filename = fs.save(image.name, image)
    solve(filename)
    path = "./media"
    path = os.path.join(path, filename)
    os.remove(path)

    return render(request, 'home.html')