import os
import matplotlib.pyplot as plt
import json
from keras.models import Model
import pickle

def createfolder():
    foldername=input("choose your network's name" )
    folderpath= str(os.getcwd())+"\ "+str(foldername)

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        print("the path is"+str(str(os.getcwd())+"\ "+str(foldername)))
        print("the folder's name is:"+str(foldername))
        return folderpath
    else:
        print("please choose a new name")
        createfolder()


def autoencoder(path, hist, encoderpart, decoderpart):
    history= open(str(path)+"/hist.pickle","wb")
    pickle.dump(hist,history)
    history.close()

    encoder=open(str(path)+"/encoder.pickle","wb")
    pickle.dump(encoderpart, encoder)
    encoder.close()

    decoder=open(str(path)+"/decoder.pickle", "wb")
    pickle.dump(decoderpart,decoder)
    decoder.close()

def saveplot(path,name):
    plt.savefig(str(path)+"/"+str(name))
    print("plot saved: "+str(name))

def savenetwork(path,Model,descriptor):
    networklocation = open(str(path) + "/"+str(descriptor)+".pickle", "wb")
    pickle.dump(Model, networklocation)
    networklocation.close()


