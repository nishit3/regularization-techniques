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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset.tensors[0]))


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


l1lambda_values = np.linspace(0, 0.55, 10)
train_accuracies = []
test_accuracies = []
total_weights = 0

for i, l1lambda_val in enumerate(l1lambda_values):

    classifier = MultiClassClassifier()
    if i == 0:
        for name, prop in classifier.named_parameters():
            if 'bias' not in name:
                total_weights += torch.numel(prop)
        print(total_weights)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=classifier.parameters(), lr=learning_rate)
    train_per_epoch = []
    test_per_epoch = []

    for epoch_i in range(1000):
        batch_acc = []

        for X, y in train_loader:
            pred_labels = classifier(X)
            loss = loss_fun(pred_labels, y)

            l1_term = torch.tensor(0., requires_grad=True)
            for name, prop in classifier.named_parameters():
                if 'bias' not in name:
                    l1_term = l1_term + torch.sum(torch.abs(prop))

            loss += l1lambda_val*l1_term/total_weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_labels = torch.argmax(pred_labels, axis=1)
            batch_acc.append(100*torch.mean((pred_labels == y).float()).item())

        train_per_epoch.append(np.mean(batch_acc))
        classifier.eval()
        test_X, test_y = next(iter(train_loader))
        test_predictions = classifier(test_X)
        test_predictions = torch.argmax(test_predictions, axis=1)
        test_per_epoch.append(100*torch.mean((test_predictions == test_y).float()).item())
        classifier.train()

    train_accuracies.append(np.mean(train_per_epoch))
    test_accuracies.append(np.mean(test_per_epoch))

plt.plot(l1lambda_values, train_accuracies, label='train')
plt.plot(l1lambda_values, test_accuracies, label='test')
plt.legend()
plt.xlabel("L1 regularization parameter (λ)")
plt.ylabel("Accuracy %")
plt.show()
