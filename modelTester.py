import os
from data_processing import load_data, folder
from network2 import Resnet50
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import numpy as np
import torch
from tqdm import tqdm


def testModel(inp):
    accuracy=[]
    print(os.listdir())
    for i in range(10):
        train, test, _, _ = load_data(
            train_batch_size=256, test_batch_size=4096, train_size=3500, 
            n_crv=6, noise_mag=0.1*i, noise_mag_train=0.3, mask_sr=False, test_size=1000, inp=inp)
        model = Resnet50(1e-3,1e-2)
        model.model.load_state_dict(torch.load('./trainedModels/'+inp+"/"+inp,map_location=torch.device('cpu')))
        target, pred = model.test(test, epoch=1)
        accuracy.append(accuracy_score(target, pred))
    return accuracy 

if __name__ == '__main__':
    for i in range(10):
        train, test, _, _ = load_data(
            train_batch_size=256, test_batch_size=4096, train_size=3500, 
            n_crv=6, noise_mag=0.1*i, noise_mag_train=0.3, mask_sr=False, test_size=1000)
        model = Resnet50()
        model.model.load_state_dict(torch.load("resnet18-0.pth",map_location=torch.device('cpu')))
        target, pred = model.test(test, epoch=1)
        print('Accuracy: {:.6f}'.format(accuracy_score(target, pred)))
        print(confusion_matrix(target,pred))
        #print(confusion_matrix(target,pred))
