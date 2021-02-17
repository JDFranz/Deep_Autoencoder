import tensorflow as tf
import numpy as np
import komm
import keras as ks
from keras import models
from keras.layers import Conv2D, Input, Conv2DTranspose
from numpy.core._multiarray_umath import dtype
import matplotlib.pyplot as plt
import layer
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import savetraining
import PIL
from PIL import Image
import pickle


#loading data
def loadimg(path="e9f3f74b-07f9-4374-8df9-422910fcbd3c.jpg"):
    img=Image.open((path))
    img=img.resize((200,200))
    imgasarray=np.asarray(img)
    return imgasarray


def recallnetwork():
    from layer import  AWGNlayer
    global autencoder
    name=input("which network shall be recalled? ")
    path=str(name)
    autencoderfile = open(path + "/testnetwork.pickle", 'rb')
    autencoder = pickle.load(autencoderfile)
    return autencoder


    #showing general information
    print(str(name) + " autencoder:")


loadimg()
model=recallnetwork()
transmission_img=model.predict()

