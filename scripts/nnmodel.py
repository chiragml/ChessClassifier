#!/bin/env python3
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm  # progress bar


def repl():
    import code
    code.InteractiveConsole(locals=globals()).interact()


class BoardsDataset(Dataset):
    def __init__(self, file):
        self.boards = pickle.load(open(file, 'rb'))
        self.wc = len(self.boards['win'])
        self.lc = len(self.boards['loss'])
        # self.dc = len(self.boards['draw'])

    def __len__(self):
        return self.wc + self.lc

    def __getitem__(self, idx):
        win = idx < self.wc
        k, i = ('win', idx) if win else ('loss', idx - self.wc)
        # print(self.wc, self.lc, idx, win, k, i)
        x, y = self.boards[k][i], win
        # return self.boards[k][i], win
        x = torch.tensor(x)
        x = x.to(torch.float32)
        y = torch.tensor(y)
        y = y.type(torch.LongTensor)
        # print('huh', y, y.dtype)
        return x, y


device = "cuda" if torch.cuda.is_available() else "cpu"


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 2),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


train_dataloader = DataLoader(BoardsDataset(
    'otb.heuristics.pkl'), batch_size=64, shuffle=True)


def train(dataloader, model, loss_fn, optimizer):
    print("training")
    # size = len(dataloader.dataset)
    model.train()
    for (X, y) in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn):
    print("testing")
    size = len(dataloader.dataset)
    batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= batches
    correct /= size
    print(f"accuracy: {correct}, loss: {test_loss}")


model = NeuralNetwork().to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in range(5):
    train(train_dataloader, model, loss_fn, optimizer)
    test(train_dataloader, model, loss_fn)  # TODO separate test and train
