import uvicorn
from fastapi import FastAPI, Request, File, UploadFile,Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
import psycopg2
import os
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

import base64
import tensorflow
from tensorflow import keras
from PIL import Image
import pandas as pd
import numpy as np
import io

import h5py

import os
import subprocess
    

model=keras.models.load_model('model_feat2.h5')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("open.html", {"request": request})

@app.get('/index')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/login')
def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get('/sign')
def sign(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})



@app.post("/upload_image", response_class=HTMLResponse)
async def upload_image(request: Request, image_file: UploadFile = File(...)):
    file_path = "model_feat2.h5"  # Replace with the path to your HDF5 file
    
    if not os.path.isfile('model_feat2.h5'):
        subprocess.run(['curl --output model_feat2.h5 "https://github.com/pavan4679/Dr_Detection/blob/main/model_feat2.h5"'], shell=True)

    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)

    with open(save_path, "wb") as f:
        content = await image_file.read()
        f.write(content)

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Decode the predictions
    class_labels = {0: 'No_Dr', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferate_Dr'}
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    print("Predicted class:", predicted_class_label)

    
    
   
    return templates.TemplateResponse("result.html", context)



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
    