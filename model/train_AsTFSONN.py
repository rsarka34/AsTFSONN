import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!pip install mat73
import mat73
import cv2 as cvlib
import librosa
from google.colab import drive
drive.mount('/content/gdrive')
from fastonn import SelfONN2d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torchsummary import summary

sig_dict=mat73.loadmat('/content/gdrive/MyDrive/ARKA/Asthma_classification/all_sigs.mat')
sig=sig_dict['all_sig'];
sig=sig.transpose()
print(sig.shape)

labels=pd.read_excel('/content/gdrive/MyDrive/ARKA/Asthma_classification/labels.xlsx',header=None)
l=np.array(labels)
label_list=[]
for i in range (1362):
  if l[i]=='Asthma':
    label_list.append(0)
  else:
    label_list.append(1)
Y=np.array(label_list)
print(Y.shape)

r=len(sig[:,1])
nfft=1024
win_length=1024
hop_length=410;sr=4000
audio_rgb_list=[]
d_shape=64

for i in range (r):
  clip=sig[i,:]
  mel_spec=librosa.feature.melspectrogram( y=clip, sr=4000,n_mels=64, n_fft=1024, hop_length=410, win_length=1024, window='hann')
  log_spectrogram = librosa.amplitude_to_db(mel_spec)
  norm=(log_spectrogram-np.min(log_spectrogram))/(np.max(log_spectrogram)-np.min(log_spectrogram))
  img = norm
  img=cvlib.resize(img, dsize=(d_shape,d_shape), interpolation=cvlib.INTER_CUBIC)
  audio_rgb_list.append(img)

X=np.array(audio_rgb_list)
print('shape of prev spectrogram dataset'+str(np.shape(X)))
X=np.reshape(X,(1362,1,64,64))
print('shape of one spectrogram dataset'+str(np.shape(X)))

def create_datasets(X, y, test_size=0.15,seed=None):
    X_train1, X_test, y_train1, y_test = train_test_split(X, Y, test_size=0.1,random_state=seed)
    print(y_test.shape)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train1, y_train1, test_size=0.1,random_state=seed)
    X_train, X_valid, X_test = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_valid, X_test)]
    y_train, y_valid, y_test = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_valid, y_test)]

    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    test_ds= TensorDataset(X_test, y_test)
    return train_ds, valid_ds, test_ds

def create_loaders(train_ds, valid_ds, test_ds, bs, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    test_dl = DataLoader(test_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl,test_dl

print('Preparing datasets')
trn_ds, val_ds, tst_ds = create_datasets(X,Y,seed=25)
bs = 128
trn_dl, val_dl,tst_dl = create_loaders(trn_ds, val_ds,tst_ds, bs)


class ONNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 3 conv blocks / flatten / linear / softmax
        self.onnv1 = nn.Sequential(
            SelfONN2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                q=3
            ),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.onnv2 = nn.Sequential(
             SelfONN2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                q=3
            ),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.onnv3 = nn.Sequential(
            SelfONN2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                q=3
            ),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32*6*6, 500)#
        self.relu=nn.ELU()
        self.dropout =nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(500, 2)
        self.output_l = nn.Softmax()

    def forward(self, input_data):
        x = self.onnv1(input_data)
        x = self.onnv2(x)
        x = self.onnv3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x= self.dropout(x)
        logits = self.linear2(x)
        predictions = self.output_l(logits)
        return x#logits

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ONNNetwork().to(device)
summary(model, (1, 64, 64))

lr = 0.001
n_epochs = 252
iterations_per_epoch = len(trn_dl)
num_classes = 2
best_acc = 0
patience, trials = 500, 0
base = 1
step = 2
trainloss_history = []
valacc_history = []
valloss_history = []
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)


print('Start model training')

for epoch in range(1, n_epochs + 1):
    for i, (x_batch, y_batch) in enumerate(trn_dl):
        model.train()
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        opt.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        preds_tr = F.log_softmax(out_tr, dim=1).argmax(dim=1)
        total += y_batch.size(0)
        correct += (preds_tr == y_batch).sum().item()
        loss.backward()
        opt.step()
    trainloss_history.append(loss)
    train_acc = correct / total
    trainacc_history.append(train_acc)
    with torch.no_grad():
        model.eval()
        correct, total = 0, 0
        for x_val, y_val in val_dl:
            x_val, y_val = [t.cuda() for t in (x_val, y_val)]
            out = model(x_val)
            valid_loss = criterion(out, y_val)
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_val.size(0)
            correct += (preds == y_val).sum().item()
    valloss_history.append(valid_loss)
    valid_acc = correct / total
    valacc_history.append(valid_acc)
    print(f'Epoch: {epoch:3d}. Training Loss: {loss:.4f}. Validation Loss: {valid_loss:.4f}. Training Acc.: {train_acc:2.2%}  Validation Acc.: {valid_acc:2.2%}')

val_loss=torch.tensor(valloss_history,device = 'cpu'); val_loss_np=val_loss.numpy()
train_loss=torch.tensor(trainloss_history,device = 'cpu');train_loss_np=train_loss.numpy()
train_acc=torch.tensor(trainacc_history,device = 'cpu');train_acc_np=train_acc.numpy()
val_acc=torch.tensor(valacc_history,device = 'cpu'); val_acc_np=np.array(val_acc)

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(val_loss_np,'g');plt.plot(train_loss_np,'r');
plt.ylabel('Loss');plt.xlabel('Epochs')
plt.subplot(122)
plt.plot(val_acc_np,'g');plt.plot(train_acc_np,'r');
plt.ylabel('Accuracy');plt.xlabel('Epochs')
plt.ylim(0.3,0.98)

PATH= '/content/gdrive/MyDrive/ARKA/7class_disease_work/MSONN_results/MSONN_trained_on_'+str(ind_time)+'.pt'
torch.save(model, PATH)
torch.save(model.state_dict(), '/content/gdrive/MyDrive/ARKA/7class_disease_work/MSONN_results/MSONN_trained_on_'+str(ind_time)+'.pth')



