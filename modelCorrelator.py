from data_processing import load_data, folder
from network2 import Resnet50
import numpy as np

def compare(m1,m2):
    train, test, _, _ = load_data(
                train_batch_size=256, test_batch_size=4096, train_size=2300, 
                n_crv=6, noise_mag=0.1, noise_mag_train=0.1, mask_sr=False, test_size=1000)
    target1, pred1 = m1.test(test, epoch=1)
    target2, pred2 = m2.test(test, epoch=1)
    pred1=np.array(pred1)
    pred2=np.array(pred2)
    target1=np.array(target1)
    target2=np.array(target2)
    correct1=target1==pred1
    correct2=target2==pred2
    difference=np.logical_or(correct1,correct2)
    print("Overestimate:"+str(np.count_nonzero(difference)/pred1.size))

    
