import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm, trange
import pickle
import matplotlib.pyplot as plt

device = torch.device('cuda')

class PredictorModel(nn.Module):
    """Generate latency fusion fine-tuning model.
    
    A fully connected model with 16 hidden nodes.
    """
    def __init__(self, device):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(12, 16), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(16,1))
        self.to(device)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.0005)
        self._initialize_weights()

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x

    def get_loss(self, true_latency, predicts):
        """Calculate loss.
        
        Args:
            true_latency (float): ground truth.
            predicts (float): predicted value.
        
        Returns:
            Prediction loss.
        """

        cost = nn.MSELoss()
        loss = cost(predicts, true_latency)
        
        return loss
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

def data_process(path, ratio):
    """Generate data loaders for training and evaluation.
    
    Args:
        path (str): File path of raw data set.
        ratio (float): Percentage of training set.
    
    Returns:
        Training loader and evaluation loader.
    """
    df_data = pd.read_csv(path)
    enc = OneHotEncoder()
    em_op1 = enc.fit_transform(df_data[["operator1"]]).toarray()
    em_op2 = enc.transform(df_data[["operator2"]]).toarray()
    df_data = np.array(df_data)
    train_index = np.random.randint(0,df_data.shape[0],np.ceil(df_data.shape[0]*ratio).astype(int))
    f_index = np.linspace(0, df_data.shape[0]-1, df_data.shape[0]).astype(int)
    test_index = np.delete(f_index, train_index, 0)

    data = np.concatenate((em_op1, em_op2), axis=1)
    data = np.concatenate((data, df_data[:, 2:4]), axis=1).astype(np.float32)
    results = df_data[:, 4]
    scaler = preprocessing.StandardScaler()
    train_data = scaler.fit_transform(data)
    pickle.dump(scaler, open('scaler_1.pkl','wb'))
    x = torch.from_numpy(train_data).to(device).float()
    y = torch.from_numpy(results).to(device).view(results.shape[0], 1).float()
    x_train = x[train_index]
    y_train = y[train_index]
    x_test = x[test_index]
    y_test = y[test_index]

    train_set = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, drop_last=False)
    test_set = TensorDataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True, drop_last=False)
    return train_loader, test_loader

def model_train(model, train_loader, test_loader, epoch_num):
    """Train model.
    
    The loss on training set and evaluation set will be illustrated in real time.

    Args:
        model (object): Latency fusion fine-tuning model.
        train_loader: Training loader.
        test_loader: Evaluation loader.
        epoch_num (int): Training epochs.
    """
    pbar = tqdm(total=epoch_num)
    train_loss_list = []
    test_error_list = []
    title_dict = {'family':'Times New Roman', 'weight':'600', 'size':19}
    labele_dict = {'family':'Times New Roman', 'weight':'600', 'size': 15}
    plt.ion()
    figure, ax = plt.subplots(1,2, figsize=(15,5))

    ax[0].grid(linestyle='--')
    ax[1].grid(linestyle='--')
    for epoch in range(epoch_num):
        train_total_loss = 0
        train_num = 0
        for batch, (x, y) in enumerate(train_loader):
            predicts = model(x)
            loss = model.get_loss(y, predicts)
            train_total_loss = train_total_loss + loss
            train_num += 1
            model.opt.zero_grad()
            loss.backward(retain_graph=True)
            model.opt.step()
        
        test_error = 0
        test_total_loss = 0
        test_num = 0
        model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(test_loader):
                predicts = model(x)
                error = np.mean((abs(y-predicts)).cpu().numpy())
                
                test_error = test_error + error
                loss = model.get_loss(y, predicts)
                test_total_loss = test_total_loss + loss
                test_num += 1

            pbar.update(1)
            pbar.set_description("\rEpoch: {:d} batch: {:d} train_loss: {:.4f} test_loss: {:.4f}"
                  .format(epoch+1, batch+1, (train_total_loss/train_num).item(), (test_total_loss/test_num).item()))
            test_error_list.append(test_error/test_num)
            train_loss_list.append((train_total_loss/train_num).item())
            ax[1].plot(test_error_list, color='midnightblue')

            ax[0].plot(train_loss_list, color='firebrick')
            figure.canvas.draw()
            ax[1].set_title('Evaluate error', fontdict=title_dict)
            ax[1].set_xlabel('epoch', fontdict=labele_dict)
            ax[1].set_ylabel('Error (ms)', fontdict=labele_dict)
            ax[1].tick_params(direction='in')
            ax[0].set_xlabel('epoch', fontdict=labele_dict)
            ax[0].set_ylabel('Loss', fontdict=labele_dict)
            ax[0].set_title('Training loss', fontdict=title_dict)
            ax[0].tick_params(direction='in')
            figure.canvas.flush_events()

    torch.save(model.state_dict(), "fusion_predictor(2.28test).pt")

model = PredictorModel(device)
# File path of raw data set
path = "local_latency.csv"
# Percentage of training set
split_ratio = 0.8
# Training epochs
epoch_num = 60
# Generate training loader and evaluation loader
train_loader, test_loader = data_process(path, split_ratio)

model_train(model, train_loader, test_loader, epoch_num)
