from data_processing import load_data, folder
from ensembleNetwork import threeDRes
from ensembleNetwork import twoDRes
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from modelCorrelator import compare


if __name__ == '__main__':
    #outF = open("myOutFile.txt","a")
    # for i in range(10):
    #     train, test, _, _ = load_data(
    #             train_batch_size=256, test_batch_size=4096, train_size=2300, 
    #             n_crv=6, noise_mag=0.1*i, noise_mag_train=0.1*i, mask_sr=False, test_size=1000)
    #     model = Resnet50()
    #     loss, test_loss = model.fit(train, test, epoch=100)
    #     target, pred = model.test(test, epoch=1)
    #     torch.save(model.model.state_dict(), 'resnet18-{}.pth'.format(i))
    #     print('Accuracy: {:.6f}'.format(accuracy_score(target, pred)))
    #     #outF.write('Accuracy: {:.6f}'.format(accuracy_score(target,pred)))
    #     print(confusion_matrix(target,pred))

    train, test, _, _ = load_data(
            train_batch_size=256, test_batch_size=4096, train_size=2300, 
            n_crv=6, noise_mag=0.15, noise_mag_train=0.15, mask_sr=False, test_size=1000)
    model1=twoDRes()
    model2=threeDRes()
    model1.model.load_state_dict(torch.load("resnet18-0.pth",map_location=torch.device('cpu')))
    model2.model.load_state_dict(torch.load("resnet18-1.pth",map_location=torch.device('cpu')))
    target1, pred1 = model1.test(test, epoch=1)
    target2, pred2 = model2.test(test, epoch=1)
    pred1=np.asarray(pred1)
    pred2=np.asarray(pred2)
    pred=(pred1+pred2)
    pred=torch.tensor(pred)
    pred1=torch.tensor(pred1)
    pred2=torch.tensor(pred2)
    pred=torch.argmax(pred, dim=1)
    pred1=torch.argmax(pred1, dim=1)
    pred2=torch.argmax(pred2, dim=1)
    print('Accuracy: {:.6f}'.format(accuracy_score(target1, pred)))
    print('Accuracy: {:.6f}'.format(accuracy_score(target1, pred1)))
    print('Accuracy: {:.6f}'.format(accuracy_score(target2, pred2)))
    
    