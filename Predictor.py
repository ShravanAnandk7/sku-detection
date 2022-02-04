"""
Users are free to copy and distribute only with citation.
https://github.com/ShravanAnandk7/Keras-Image-Embeddings-using-Contrastive-Loss
Last updated 09 Jan 2022
"""
#%%
import os
import numpy as np
import pandas as pd
from functools import partial
from cv2 import cv2
import json
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.backend as K
BASE_DIR    = os.path.dirname(__file__)
os.chdir(BASE_DIR)
# PARAMTERS
MODEL_DIR       =  os.path.join(BASE_DIR,"models")
DATASET_DIR     =  os.path.join(BASE_DIR,"datasets")
INPUT_SHAPE     =  299
EMBEDDING_SIZE  =  32
#%% Build network model
def pre_process(image):
    # image = cv2.resize(image,(INPUT_SHAPE, INPUT_SHAPE))
    image = image/127.5 -1
    return image
def euclidean_distance(x,y):
    """
    Euclidean distance metric
    """
    return np.sum(np.square(x-y), axis=-1)
def base_network():
    """
    Base CNN model trained for embedding extraction
    """
    return( 
            KM.Sequential(
                [   
                    KL.Input(shape=(INPUT_SHAPE,INPUT_SHAPE,3)),
                    KL.Conv2D(8,(3,3)),
                    KL.ReLU(),
                    KL.MaxPool2D(pool_size=(1,2)),
                    # KL.BatchNormalization(),
                    KL.Conv2D(16,(3,3)),
                    KL.ReLU(),
                    KL.MaxPool2D(pool_size=(2,1)),
                    KL.BatchNormalization(),
                    KL.Conv2D(32,(3,3)),
                    KL.ReLU(),
                    KL.MaxPool2D(pool_size=(1,1)),
                    KL.GlobalAveragePooling2D(),
                    # Don't Change the below layers
                    KL.Dense(EMBEDDING_SIZE,activation = 'relu'),
                    KL.Lambda(lambda x: K.l2_normalize(x,axis=-1))
                ]))
model = base_network()
model.load_weights(os.path.join(MODEL_DIR, "few-shot.h5"))
#%% Few-Shot model embedding prediction
input = pre_process(cv2.imread(os.path.join(BASE_DIR,"predict_image.jpg")))
output_embeddings = model.predict(np.expand_dims(input,axis=0))[0]
with open("few-shot-emedding.json",'r') as json_file:
    embedding_dict = json.load(json_file)
prediction_dict ={}
for object, embedding in embedding_dict.items():
    prediction_dict[object] = euclidean_distance( np.array(embedding) ,output_embeddings)
top_predictions = sorted(prediction_dict.items(),key = lambda x: x[1])[:5]
print("Top Predictions \n", top_predictions)
#%% One-Shot model embedding prediction
input = pre_process(cv2.imread(os.path.join(BASE_DIR,"predict_image.jpg")))
output_embeddings = model.predict(np.expand_dims(input,axis=0))[0]
with open("one-shot-emedding.json",'r') as json_file:
    embedding_dict = json.load(json_file)
prediction_dict ={}
for object, embedding in embedding_dict.items():
    prediction_dict[object] = euclidean_distance( np.array(embedding) ,output_embeddings)
top_predictions = sorted(prediction_dict.items(),key = lambda x: x[1])[:5]
print("Top Predictions \n", top_predictions)
