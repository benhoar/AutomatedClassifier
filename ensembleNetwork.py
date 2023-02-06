from twodresnet import resnet18 as twoD
from resnet import resnet18 as threeD
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt

class twoDRes:
    def __init__(self, device=('cuda' if torch.cuda.is_available() else 'cpu')) -> None:
        self.device = device
        self.model = twoD(num_classes=9)
        self.model = self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(),weight_decay=1e-5)
    
    def fit(self, train_loader, test_loader, epoch=100):
        self.model.train()
        loss_list = []
        test_loss = []
        for _ in tqdm(range(epoch)):
            train_loss = 0
            size=0
            for batch, data in enumerate(train_loader):
                X, y = data['data'].to(device=self.device), data['label'].to(device=self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                classes = torch.argmax(pred, dim=1)
                train_loss += (classes != y).float().sum()/y.size(dim=0)
            loss_list.append(train_loss)
            target, pred = self.test(test_loader, epoch=1)
            test_loss.append(accuracy_score(target,pred))
        return loss_list, test_loss
    
    def test(self, test_loader, epoch=1):
        self.model.eval()
        self.target = []
        self.prediction = []
        with torch.no_grad():
            for _ in range(epoch):
            #for _ in tqdm(range(epoch)):
                for batch, data in enumerate(test_loader):
                    X, y = data['data'].to(device=self.device), data['label'].to(device=self.device)
                    pred = self.model(X).data
                    self.target += y.tolist()
                    self.prediction += pred.detach().tolist()
#         self.cm = confusion_matrix(self.target, self.prediction, normalize='true')
#         self.acc = accuracy_score(self.target, self.prediction)
        return self.target, self.prediction
    
class threeDRes:
    def __init__(self, device=('cuda' if torch.cuda.is_available() else 'cpu')) -> None:
        self.device = device
        self.model = threeD(num_classes=9)
        self.model = self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(),weight_decay=1e-5)
    
    def fit(self, train_loader, test_loader, epoch=100):
        self.model.train()
        loss_list = []
        test_loss = []
        for _ in tqdm(range(epoch)):
            train_loss = 0
            size=0
            for batch, data in enumerate(train_loader):
                X, y = data['data'].to(device=self.device), data['label'].to(device=self.device)
                X=torch.unsqueeze(X,dim=1)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                classes = torch.argmax(pred, dim=1)
                train_loss += (classes != y).float().sum()/y.size(dim=0)
            loss_list.append(train_loss)
            target, pred = self.test(test_loader, epoch=1)
            test_loss.append(accuracy_score(target,pred))
        return loss_list, test_loss
    
    def test(self, test_loader, epoch=1):
        self.model.eval()
        self.target = []
        self.prediction = []
        with torch.no_grad():
            for _ in range(epoch):
            #for _ in tqdm(range(epoch)):
                for batch, data in enumerate(test_loader):
                    X, y = data['data'].to(device=self.device), data['label'].to(device=self.device)
                    X=torch.unsqueeze(X,dim=1)
                    pred = self.model(X).data
                    self.target += y.tolist()
                    self.prediction += pred.detach().tolist()
#         self.cm = confusion_matrix(self.target, self.prediction, normalize='true')
#         self.acc = accuracy_score(self.target, self.prediction)
        return self.target, self.prediction
    

class ResnetReg:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = resnet18(num_classes=1).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-5)

    def fit(self, train_loader, epoch=100):
        self.model.train()
        loss_list = []
        for _ in tqdm(range(epoch)):
            train_loss = 0
            for batch, data in enumerate(train_loader):
                X, y = data['data'].to(device=self.device), data['label'].to(device=self.device)
                y[y >= 0.5] = 1
                y[y < 0.5] = 0
                pred = self.model(X).view(-1)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            loss_list.append(train_loss)
        return loss_list

    def test(self, test_loader, epoch=1):
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for _ in tqdm(range(epoch)):
                for batch, data in enumerate(test_loader):
                    X, y = data['data'].to(device=self.device), data['label'].to(device=self.device)
                    pred = self.model(X).view(-1)
                    loss += self.loss_fn(pred, y).item()
        print(loss / epoch)
        return pred.cpu().detach(), y.cpu().detach()