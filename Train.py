# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 15:51:06 2022

@author: Pranav
"""

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope
import os
from data_processing import load_data, folder
from network import Resnet50 as threeDRes
from network2 import Resnet50 as twoDRes
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from modelCorrelator import compare
from modelTester import testModel

#Parameterized function for training models, returns accuracy

# BEN Train and test -> convert to percentage of available data
def findAccuracy(search_space):
    lr,wd=search_space["learning"],search_space["weight_decay"]
    model = twoDRes(lr,wd)
    train, test, _, _ = load_data(
            train_batch_size=256, test_batch_size=4096, train_size=2300, 
            n_crv=6, noise_mag=0.15, noise_mag_train=0.15, mask_sr=False, test_size=1000, inp=inp, peaknoise_mag=0)
    loss,test_loss = model.fit(train, test, epoch=100)
    target, pred = model.test(test, epoch=1)
    return accuracy_score(target, pred)

if __name__ == '__main__':
    #input prompt

    # BEN clearer directions on how to select models
    print("What mechanisms to classify:")
    inp=input("E "+"EC1 "+"CE "+"DL "+"SR "+"T "+"ECP "+"DISP "+"ECE \n")

    #check whether model is already saved in directory
    # BEN use clearer names {dirlist: done_models, i:, model, }
    dirlist = os.listdir('./trainedModels')
    flag=0
    for i in dirlist:
        if i==inp:
                flag=1
    
    #load data and initialize model
    train, test, _, _ = load_data(
            train_batch_size=256, test_batch_size=4096, train_size=2300, 
            n_crv=6, noise_mag=0.15, noise_mag_train=0.15, mask_sr=False, test_size=1000, inp=inp, peaknoise_mag=0)
    model = twoDRes(1e-3,1e-2)
    os.chdir('./trainedModels')

    #if model is new, train one, otherwise return stored model
    if flag==0:
        os.mkdir(inp)
        os.chdir('./'+inp)
        loss, test_loss = model.fit(train, test, epoch=100)
        torch.save(model.model.state_dict(), inp)
        target, pred = model.test(test, epoch=1)
        f=open("Confusion_Matrix", "x")
        f.write(str(confusion_matrix(target,pred)))
        f.close()
    else:
        os.chdir('./'+inp)
        model.model.load_state_dict(torch.load(inp,map_location=torch.device('cpu')))
        target, pred = model.test(test, epoch=1)
    print('Accuracy: {:.6f}'.format(accuracy_score(target, pred)))
    print(confusion_matrix(target,pred))
    os.chdir("../")
    os.chdir("../")

    #test with noise
    if flag==0:
        accuracy = testModel(inp)
        os.chdir("./trainedModels")
        os.chdir(inp)
        f=open("noise.txt","x")
        for i in accuracy:
            f.write(str(i))
            f.write("\n")
        f.close()

    #hyperparameter optimization space
    space = {
        "learning": hp.lognormal("learning",-3,1),
        "weight_decay": hp.lognormal("weight_decay",-2,1),
    }

    #train with hyperparameter optimization
    print(os.listdir())
    trials = Trials()
    best = fmin(
        fn=findAccuracy,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )
    print("Max possible: {:.6f}".format(best))