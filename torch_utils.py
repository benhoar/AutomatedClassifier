import pandas as pd
import os
import numpy as np
import cv2
import torch
from resnet import resnet18
from data_processing import folder
from encap_ML_Demo import treatment, reformat, prediction_treatment
from io import StringIO
import uuid

import re
# load model, for now only load one model
mdl_list = []
device = torch.device('cuda')
for k in range(8):
    m = resnet18(num_classes=5)
    m.load_state_dict(torch.load('resnet18-3-{}.pth'.format(k), map_location='cpu'))
    m.eval()
    exec('m{}=m'.format(k))
    exec('m{}.eval()'.format(k))
    exec(f'mdl_list.append(m{k})')


# data processing, only for Gamry, 6 curves. If using CHI and more curves, need more coding
def bytes_reader(bytes):
    s = str(bytes, 'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)
    trans = reformat(df)
    return trans

# predict
def ML_prediction2(read):

    uid = str(uuid.uuid4())
    read = read.to_numpy()
    scan_rate, scan_rate_idx = np.unique(read[:, 2], return_index=True)
    if len(scan_rate) == 1:
        length = 1
    else:
        length = read.shape[0] // (scan_rate_idx[1] - scan_rate_idx[0])
    mat_val = cv2.resize(np.vstack([x[:, 1] for x in np.split(read, scan_rate_idx[1:])]), (1000, length))
    mat_val = np.pad(mat_val, ((0, 6 - length), (0, 0)))
    mat_scan = np.tile(scan_rate, (500, 1)).T
    mat_scan = np.pad(mat_scan, ((0, 6 - length), (0, 0)))
    mat_val1 = mat_val[:, :500:]
    mat_val2 = mat_val[:, :-501:-1]
    X = torch.tensor(np.stack((mat_val1, mat_val2, mat_scan), axis=1)).float().unsqueeze(0)
    #output_name = os.path.join('auto', 'processed', uid + '.csv')
    label = list(folder.keys())
    df2 = pd.DataFrame()

    for i in range(8):
        model = mdl_list[i]
        model.eval()
        with torch.no_grad():
            #exec(f'Y=m{i}(X)',globals())
            Y = model(X)
            _, pred = torch.max(Y, 1)
            pred = pred.detach().tolist()
        for j, l in enumerate(pred):
            logits = Y[j].detach().cpu().numpy()
            logits -= np.max(logits)
            logits = np.exp(logits)
            logits /= np.sum(logits)
            df2 = pd.concat([df2, pd.DataFrame([uid, label[l], logits[0], logits[1], logits[2], logits[3], logits[4], i]).T], axis=0)

    df2.columns = ['File Name', 'Predicted Label', 'CE', 'EC', 'E', 'ECE', 'DISP', 'idx']

    return df2


def final_result(temp_data):
    fin_df = prediction_treatment(temp_data)
    return fin_df

# start from the CSV file, then reformat, and ML prediction







