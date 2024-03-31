# dynamic.py
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Annotated
from pydantic import BaseModel
import base64
import tensorflow
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import os
from collections import Counter
from google.cloud import bigquery, storage
from google.oauth2 import service_account
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input



app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


UPLOAD_FOLDER='static'

class Item(BaseModel):
    image_Path : str | None = None

@app.get("/")
async def dynamic_file(request: Request):
    path = "No Image Uploaded Yet"
    prediction = [[0]]
    return templates.TemplateResponse("index.html", {"request": request, "img_Path": path ,"probability": prediction})


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



@app.post("/upload_image")
async def upload_image(request: Request, image_file: UploadFile = File(...)):
    
  
    save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)

    with open(save_path, "wb") as f:
        content = await image_file.read()
        f.write(content)


    bucket_name = "sandeep_personal"
    models = ["ResNet2_Model.h5","VGG16_Model.h5"]
     
    key_path = "ck-eams-9260619158c0.json"
    client = storage.Client.from_service_account_json(key_path)


    # Retrieve the bucket
    bucket = client.get_bucket(bucket_name)


    all_predicted_class_labels = []

    for model_file in models:
        blob = bucket.blob(model_file)
        blob.download_to_filename(model_file)

        model = keras.models.load_model(model_file)
        
         
        img = image.load_img(save_path, target_size=(224, 224))  # Change image_path to save_path
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

        all_predicted_class_labels.append(predicted_class_label)
 
        os.remove(model_file)
    
    all_predicted_class_labels = np.array(all_predicted_class_labels)

    print("All predicted class labels:", all_predicted_class_labels)
    
    label_counts = Counter(all_predicted_class_labels)
    most_common_label = label_counts.most_common(1)[0][0]

    print(most_common_label)

    context = {
        "request": request,
        "predicted_class_label":most_common_label
    }


    return templates.TemplateResponse("result.html",context)

