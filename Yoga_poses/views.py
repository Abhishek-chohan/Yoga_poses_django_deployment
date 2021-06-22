import numpy as np
import tensorflow as tf

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image




img_height, img_width=150,150



model=load_model('model/model.h5')



def index(request):
    context={'a':1}
    return render(request,'index.html',context)



## output conversion function
# def out_conversion(out_array):
#     dict = {1: 'Downdog', 2: 'Goddess', 3: 'Plank', 4: 'Tree', 5: 'Warrior2'}
#     for i in range(5):
#         if out_array[i] == 1.:
#             return dict[i+1]

def predict_pose(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x=x/255
    x=x.reshape(1,img_height, img_width,3)
    prediction=model.predict(x)
    dict = {0: 'Downdog', 1: 'Goddess', 2: 'Plank', 3: 'Tree', 4: 'Warrior2'}

    predictedLabel= dict[np.argmax(prediction)]

    context={'filePathName':filePathName,'predictedLabel':predictedLabel}
    return render(request,'index.html',context)

def view_image_library(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'view_im.html',context)