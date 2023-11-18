import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


totalDataPointsPerCluster = 200
theta = np.linspace(0, 4 * np.pi, totalDataPointsPerCluster)
r1 = 10
r2 = 17

class1 = [r1 * np.cos(theta) + np.random.randn(totalDataPointsPerCluster) * 3,
          r1 * np.sin(theta) + np.random.randn(totalDataPointsPerCluster)]

class2 = [r2 * np.cos(theta) + np.random.randn(totalDataPointsPerCluster) * 3,
          r2 * np.sin(theta) + np.random.randn(totalDataPointsPerCluster)]

data = np.hstack((class1, class2)).T
labels = np.vstack((np.zeros((totalDataPointsPerCluster, 1)), np.ones((totalDataPointsPerCluster, 1))))

data = torch.tensor(data).float()
labels = torch.tensor(labels).float()

data_train, data_test, label_train, label_test = train_test_split(data, labels, train_size=0.8)

training_data_set = TensorDataset(data_train, label_train)
testing_data_set = TensorDataset(data_test, label_test)

train_loader = DataLoader(training_data_set, batch_size=16, shuffle=True)
test_loader = DataLoader(testing_data_set, batch_size=len(testing_data_set.tensors[0]))

d_rate = 0.5


class ANNClassifier(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.input = nn.Linear(2, 128)
        self.hidden = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.relu(self.input(x))
        x = nn.functional.dropout(x, p=self.drop_rate, training=self.training)
        x = nn.functional.relu(self.hidden(x))
        x = nn.functional.dropout(x, p=self.drop_rate, training=self.training)
        return self.output(x)


d_rates = np.linspace(0, 1, 20)
train_accuracy_wrt_drate = []
test_accuracy_wrt_drate = []

for i, d_rate in enumerate(d_rates):

    classifier = ANNClassifier(d_rate)
    loss_fun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=classifier.parameters(), lr=.01)
    train_accuracies = []
    test_accuracies = []

    for epoch_i in range(2000):
        classifier.train()
        batch_accuracies = []
        for X, y in train_loader:
            pred_labels = classifier(X)
            loss = loss_fun(pred_labels, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_accuracies.append(100 * torch.mean(((pred_labels > 0) == y).float()).item())

        train_accuracies.append(np.mean(batch_accuracies))
        classifier.eval()
        X, y = next(iter(test_loader))
        test_results = classifier(X)
        test_accuracies.append(100 * torch.mean(((test_results > 0) == y).float()).item())

    train_accuracy_wrt_drate.append(np.mean(train_accuracies))
    test_accuracy_wrt_drate.append(np.mean(test_accuracies))

plt.plot(d_rates, train_accuracy_wrt_drate, label="train")
plt.plot(d_rates, test_accuracy_wrt_drate, label="test")
plt.legend()
plt.xlabel("Dropout Probability")
plt.ylabel("Accuracy %")
plt.show()
