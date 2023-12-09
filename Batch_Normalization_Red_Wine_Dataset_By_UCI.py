import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = pd.read_csv('winequality-red.csv', sep=';')
data = data[data['total sulfur dioxide'] < 200]
data['isGood'] = (data['quality'] > 5)

data = data.drop(columns=["quality"])

indpndt_vars = data.keys()
indpndt_vars = indpndt_vars.drop('isGood')

feature_matrix = data.iloc[:, :-1].values
target = data.iloc[:, -1].values
target = np.reshape(target, (-1, 1))


train_X, test_X, train_y, test_y = train_test_split(feature_matrix, target, train_size=0.8)
train_X = torch.tensor(train_X).float()
test_X = torch.tensor(test_X).float()
train_y = torch.tensor(train_y).float()
test_y = torch.tensor(test_y).float()


train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)


class BinaryWineTasteClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(len(indpndt_vars), 16)
        self.hidden1 = nn.Linear(16, 32)
        self.batchNorm1 = nn.BatchNorm1d(16)
        self.hidden2 = nn.Linear(32, 32)
        self.batchNorm2 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 1)

    def forward(self, x, doBatchNorm):
        x = nn.functional.relu(self.input(x))

        if doBatchNorm:
            x = nn.functional.relu(self.hidden1(self.batchNorm1(x)))
            x = nn.functional.relu(self.hidden2(self.batchNorm2(x)))

        else:
            x = nn.functional.relu(self.hidden1(x))
            x = nn.functional.relu(self.hidden2(x))

        return self.output(x)


classifier = BinaryWineTasteClassifier()

loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=.01)

epochs = np.linspace(1, 1000, num=1000)
# batch_sizes = 2**np.arange(2, 10, 2)
batch_sizes = [16]

results = list(np.zeros((len(batch_sizes), len(epochs))))

for batch_i in range(len(batch_sizes)):
    train_loader = DataLoader(train_dataset, batch_size=int(batch_sizes[batch_i]), drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(torch.detach(test_X)))

    for epoch_i, epoch in enumerate(epochs):
        for X, y in train_loader:
            predictions = classifier(X, False)
            loss = loss_func(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        classifier.eval()
        test_X, test_y = next(iter(test_loader))
        with torch.no_grad():
            pred = classifier(test_X, False)
        test_loss = loss_func(pred, test_y)
        results[batch_i][epoch_i] = torch.mean(((pred > 0) == test_y).float()).item()*100.00
        classifier.train()

plt.title("Without Batch Normalization")
for batch_i in range(len(batch_sizes)):
    plt.plot(epochs, results[batch_i], label=str(batch_sizes[batch_i]))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")
    plt.legend()
plt.show()
