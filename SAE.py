import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import torch.optim
from torch.utils.data import Dataset, DataLoader
import random
import torch.functional as F
from pre_processing import preprocessing
import matplotlib.pyplot as plt


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


class SAE(nn.Module):
    def __init__(self, input_dim, output_dim, pre = True, encoder1=128, encoder2=32, path = None):
        super().__init__()
        self.pre = pre
        self.input_layer = nn.Linear(in_features=input_dim, out_features=encoder1, bias=True)
        self.encoder1 = nn.Linear(in_features=encoder1, out_features=encoder2, bias=True)
        self.encoder2 = nn.Linear(in_features=encoder2, out_features=output_dim, bias=True)
        # self.encoder_hidden_layer = nn.Sequential(
        #     nn.Linear(in_features=input_dim, out_features=encoder1, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=encoder1, out_features=encoder2, bias=True), nn.ReLU())
        # self.encoder_output_layer = nn.Linear(
        #     in_features=encoder2, out_features=output_dim
        # )
        if pre:
            self.decoder_hidden_layer = nn.Sequential(
                nn.Linear(in_features=output_dim, out_features=encoder2, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=encoder2, out_features=encoder1, bias=True),
                nn.ReLU()
            )
            self.decoder_output_layer = nn.Linear(
                in_features=encoder1, out_features=input_dim
            )

        self.apply(weights_init)
        if path:
            current_model = self.state_dict()
            update_encoder = torch.load(path)
            new_dict = {k: v for k, v in update_encoder.items() if k in current_model.keys()}
            current_model.update(new_dict)
            self.load_state_dict(current_model)

    def forward(self, features):
        activation = torch.relu(self.input_layer(features))
        activation = torch.relu(self.encoder1(activation))
        code = self.encoder2(activation)
        if self.pre:
            activation = self.decoder_hidden_layer(code)
            reconstructed = self.decoder_output_layer(activation)
            return reconstructed, code
        else:
            value = torch.log_softmax(code, dim=1)
            return value

class Train:
    def __init__(self, X_train, y_train, X_test, y_test, pre = True, path = None):
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train)
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.FloatTensor(y_test)
        self.class_num = len(np.unique(self.y_train))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pre = pre
        self.model = SAE(input_dim=self.X_train.shape[1], output_dim=self.class_num, pre=self.pre, path = path).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        if self.pre:
            self.criterion = nn.MSELoss()

    def spilt(self, ratio):
        index = np.arange(0, len(self.X_train))
        random.shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = self.y_train[index]

        num_of_train = int(ratio*len(self.X_train))
        return self.X_train[0:num_of_train], self.y_train[0:num_of_train], self.X_train[num_of_train:], self.y_train[num_of_train:]

    def random_shuffle(self, x, y):
        index = np.arange(0, len(x))
        random.shuffle(index)
        return x[index], y[index]

    def train(self, epoch = 200, batchsize = 50):
        X_train, y_train, X_val, y_val = self.spilt(0.8)
        epoch_loss = []
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        if self.pre:
            for i in range(epoch):
                x, y = self.random_shuffle(X_train, y_train)
                round = len(x) // batchsize
                loss = 0
                for idx in range(round):
                    regulazation_loss = 0
                    train, label = x[idx*batchsize: (idx+1)*batchsize], y[idx*batchsize: (idx+1)*batchsize]
                    train, label = train.to(self.device), label.to(self.device)
                    outputs, code = self.model(train)
                    for param in self.model.parameters():
                        regulazation_loss+=torch.sum(torch.abs(param))
                    train_loss = self.criterion(outputs, train) + 0.0001*regulazation_loss
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    self.optimizer.step()
                    loss+=train_loss.item()
                val_output, val_code = self.model(X_val)
                val_loss = self.criterion(val_output, X_val)

                epoch_loss.append(val_loss.item())
            plt.plot(epoch_loss)
            plt.show()
    def get_para(self):
        p = {}
        for name, para in self.model.named_parameters():
            if name.find('encoder') != -1:
                p[name] = para
        torch.save(p, "encoder_para.pth")
        return "encoder_para.pth"

if __name__ == '__main__':
    dataset = preprocessing('DataSetA.mat')
    X_train, X_test, y_train, y_test = dataset.Data_Preprocess(slice_size=2400, L=2400, window=False)
    # print(y_train)
    train = Train(X_train, y_train, X_test, y_test)
    train.get_para()




