import os
import numpy as np
import cv2
import torch
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random

length = 1000
path = './Data2/'
folder = {
    'CE': path+'CE', 'EC1': path+'EC1', 'E': path+'E',
    'ECE': path+'ECE', 'DISP': path+'DISP', 'DL': path+'DL','ECP': path+'ECP', 'T': path+'T'
    #'CE': path+'CE/', 'EC': path+'EC/', 'E': path+'E/',
    #'ECE': path+'ECE/', 'DISP': path+'DISP/'
}

if __name__ == "__main__":

    #iterate through data folders
    for j, e in enumerate(folder.keys()):
        dirlist = os.listdir(folder[e])
        dataset = []
        scanlist=[]

        #input from each file and process
        print('Now Handling {} with {} samples'.format(e, len(dirlist)))
        for f in tqdm(dirlist):
            name, ext = os.path.splitext(f)
            if ext != '.txt':
                continue
            #print(f)
            read = np.genfromtxt(folder[e] + f, skip_header=1, delimiter=',')
            # if e == 'SR':
            #     read=read[:,1:]
            scan_rate, scan_rate_idx = np.unique(read[:, 2], return_index=True)
            print(scan_rate)
            a=np.split(read, scan_rate_idx[1:])
            for i in range(len(a)):
                a[i]=(a[i])[:2000]
            mat_val = cv2.resize(
                np.vstack([x[:, 1] for x in a]), (length, scan_rate.size))
            mat_scan = np.tile(scan_rate, (length // 2, 1)).T
            mat_val1 = mat_val[:, :length // 2]
            mat_val2 = mat_val[:, :-(length // 2 + 1):-1]
            mat = np.stack((mat_val1, mat_val2, mat_scan), axis=1)
            mat=torch.tensor(mat,dtype=torch.float32)
            if (mat.size(dim=0)<6):
                mat=torch.nn.functional.pad(mat,(0,0,0,0,6-mat.size(dim=0),0))
            #mat = np.stack((mat_val1, mat_val2), axis=1)
            dataset.append({'data': 
                mat, 'label': e, 'file': f, 'key': j})
        print(scanlist)
        with open(path+'{}.pkl'.format(e), 'wb') as f:
            pickle.dump(dataset, f)


class Data(Dataset):
    def __init__(self, array, train=False, n_crv=6, noise_mag=1, mask_sr=False, peaknoise_mag=0):
        # dataformat: 6 * n * 1000
        self.data = array
        self.train = train
        self.n_crv = n_crv
        self.noise_mag = noise_mag
        self.mask_sr = mask_sr
        self.peaknoise_mag=peaknoise_mag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
#         ncrv = np.random.randint(2, 7) if self.train else self.n_crv
        ncrv = self.n_crv
        entry = self.data[idx]['data']
        entry = entry[:,(0,1,2),:]
        peak = np.argmax(entry[:,(0,1),:], axis=0)
        valley = np.argmin(entry[:,(0,1),:], axis=0)
        mask=torch.zeros_like(entry)
        mask[max(peak[0][0]-50,0):min(peak[0][0]+50,500),(0,1),:]+=1
        mask[max(valley[0][0]-50,0):min(valley[0][0]+50,500),(0,1),:]+=1
        noise = torch.zeros_like(entry)
        noise[:, (0,1), :]=torch.normal(noise[:, (0,1), :], std=self.noise_mag)
        peaknoise=torch.zeros_like(entry)
        # peaknoise[:, (0,1), :]=torch.normal(peaknoise[:, (0,1), :], std=self.peaknoise_mag)
        # peaknoise=peaknoise*mask
        label = self.data[idx]['key']
        mask=torch.ones_like(entry)
        """ mask = torch.zeros_like(entry)
        if ncrv == 1:
            mask[np.random.randint(6)] = 1
        else:
            mask[0] = 1
            mask[5] = 1
            arr = np.random.choice([1, 2, 3, 4], size=(ncrv - 2), replace=False)
            mask[arr] = 1
        if self.mask_sr:
            mask[:, 2, :] = 0 """
        if self.train:
            return {'data': ((noise + peaknoise + entry) * mask), 'label': label}
        else:
            return {'data': ((noise + peaknoise + entry) * mask), 'label': label}


data_files = [path+'EC1.pkl', path+'E.pkl',
              path+'CE.pkl', path+'ECE.pkl', path+'DISP.pkl', path+'DL.pkl', path+'T.pkl', path+'ECP.pkl']


def load_data(train_batch_size, test_batch_size, train_size, n_crv, noise_mag, noise_mag_train, mask_sr, test_size, inp, peaknoise_mag=0):
    train, test = [], []
    inp=inp.split(',')
    for file in inp:
        train_this, test_this = train_test_split(
            pickle.load(open(path+file+".pkl", 'rb')), train_size=train_size,test_size=test_size)
        train += train_this
        test += test_this
    train_data = Data(train, train=True, n_crv=n_crv, noise_mag=noise_mag_train, mask_sr=mask_sr,peaknoise_mag=peaknoise_mag)
    test_data = Data(test, n_crv=n_crv, noise_mag=noise_mag, mask_sr=mask_sr,peaknoise_mag=peaknoise_mag)
    train_loader = DataLoader(
        train_data, batch_size=train_batch_size, num_workers=8, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)
#     print('Training {}, Testing {}'.format(len(train_data), len(test_data)))
    return train_loader, test_loader, train_data, test_data
