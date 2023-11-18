import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


iris = sbn.load_dataset("iris")
data = torch.tensor(iris[iris.columns[0:4]].to_numpy()).float()
labels = torch.zeros(len(data)).long()
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2
learning_rate = .001
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset.tensors[0]))
epochs = np.linspace(1, 1000, 1000)


class MultiClassClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4, 64)
        self.hidden = nn.Linear(64, 64)
        self.output = nn.Linear(64, 3)

    def forward(self, x):
        x = self.input(x)
        x = f.relu(x)
        x = self.hidden(x)
        x = f.relu(x)
        return self.output(x)


train_accuracies = np.zeros((6, 1000))
test_accuracies = np.zeros((6, 1000))


for i in range(6):
    train_loader = DataLoader(train_dataset, batch_size=pow(2, i+1), shuffle=True, drop_last=True)
    classifier = MultiClassClassifier()
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=classifier.parameters(), lr=learning_rate)

    for epoch_i in range(1000):
        batch_acc = []

        for X, y in train_loader:
            pred_labels = classifier(X)
            loss = loss_fun(pred_labels, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_labels = torch.argmax(pred_labels, axis=1)
            batch_acc.append(100*torch.mean((pred_labels == y).float()).item())

        train_accuracies[i][epoch_i] = np.mean(batch_acc)
        classifier.eval()
        test_X, test_y = next(iter(train_loader))
        test_predictions = classifier(test_X)
        test_predictions = torch.argmax(test_predictions, axis=1)
        test_accuracies[i][epoch_i] = 100*torch.mean((test_predictions == test_y).float()).item()
        classifier.train()


for i in range(6):
    batch_size = pow(2, i+1)
    plt.plot(epochs, train_accuracies[i], label=f'batch size = {batch_size}')
plt.title("Training")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.show()

for i in range(6):
    batch_size = pow(2, i+1)
    plt.plot(epochs, test_accuracies[i], label=f'batch size = {batch_size}')
plt.title("Testing")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.show()
