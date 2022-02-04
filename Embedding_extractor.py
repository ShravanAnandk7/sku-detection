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
import tensorflow as tf
import tensorflow.keras.utils as KU
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.losses as KLo
import tensorflow.keras.optimizers as KO
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from imgaug import augmenters as arg
BASE_DIR    = os.path.dirname(__file__)
os.chdir(BASE_DIR)
# PARAMTERS
MODEL_DIR       =  os.path.join(BASE_DIR,"models")
DATASET_DIR     =  os.path.join(BASE_DIR,"datasets")
INPUT_SHAPE     =  299
EMBEDDING_SIZE  =  32
#%% Build network model
def pre_process(image):
    image = cv2.resize(image,(INPUT_SHAPE, INPUT_SHAPE))
    image = image/127.5 -1
    return image
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
#%% Few-Shot images embedding generation
embedding_dict={}
path = os.path.join(DATASET_DIR,"few-shot-dataset")
for object in os.listdir(os.path.join(DATASET_DIR,"few-shot-dataset")):
    embedding_array = []
    for image in os.listdir(os.path.join(DATASET_DIR,"few-shot-dataset",object)):
        input = pre_process(cv2.imread(os.path.join(DATASET_DIR,"few-shot-dataset",object,image)))
        output_embeddings = model.predict(np.expand_dims(input,axis=0))
        embedding_array.append(output_embeddings[0])
    
    embedding_array = np.array(embedding_array)
    embedding_array = np.mean(embedding_array,axis=0)
    embedding_dict[object] = embedding_array.tolist()
with open("few-shot-emedding.json",'w') as json_file:
    json.dump(embedding_dict,json_file)
#%% One-Shot images embedding generation
embedding_dict={}
path = os.path.join(DATASET_DIR,"one-shot-dataset")
for image in os.listdir(os.path.join(DATASET_DIR,"one-shot-dataset")):
    input = pre_process(cv2.imread(os.path.join(DATASET_DIR,"one-shot-dataset",image)))
    output_embeddings = model.predict(np.expand_dims(input,axis=0))
    embedding_dict[image[:-4]] = output_embeddings[0].tolist()
with open("one-shot-emedding.json",'w') as json_file:
    json.dump(embedding_dict,json_file)
