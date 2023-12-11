import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

total_elements_per_class = 100

starting_xandy_coordinates_class1 = [1, 1]
starting_xandy_coordinates_class2 = [5, 1]
starting_xandy_coordinates_class3 = [1, 5]

class1 = [starting_xandy_coordinates_class1[0] + np.random.randn(total_elements_per_class), starting_xandy_coordinates_class1[1] + np.random.randn(total_elements_per_class)]
class2 = [starting_xandy_coordinates_class2[0] + np.random.randn(total_elements_per_class), starting_xandy_coordinates_class2[1] + np.random.randn(total_elements_per_class)]
class3 = [starting_xandy_coordinates_class3[0] + np.random.randn(total_elements_per_class), starting_xandy_coordinates_class3[1] + np.random.randn(total_elements_per_class)]
data_np = np.hstack((class1, class2, class3)).T

labels_np = np.concatenate((np.zeros((total_elements_per_class, 1)), np.ones((total_elements_per_class, 1)), np.ones((total_elements_per_class, 1))+1))
labels_np = labels_np.reshape(-1)
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).long()
lossFun = nn.CrossEntropyLoss()

lr_rates = np.linspace(start=0.0001, stop=0.1, num=20)
optimizers = ["SGD", "RMSprop", "Adam"]
results = np.zeros((len(optimizers), len(lr_rates)))
no_of_experiments = 20

for experimentNum in range(no_of_experiments):
    for opt_i, optimizer in enumerate(optimizers):
        for lr_i, lr in enumerate(lr_rates):

            ANN_classifier = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 3),
                nn.Softmax(dim=1)
            )

            optmzr = getattr(torch.optim, optimizer)
            optmzr = optmzr(ANN_classifier.parameters(), lr)
            epochs = 1000

            resultant_accuracy = 0
            for epoch in range(epochs):
                class_predictions = ANN_classifier(data)
                loss = lossFun(class_predictions, labels)
                optmzr.zero_grad()
                loss.backward()
                optmzr.step()

                if epoch > epochs-(10+1):
                    outputs_for_data = ANN_classifier(data)
                    result = torch.argmax(outputs_for_data, 1)
                    result = result[result == labels]
                    accuracy = len(result) / len(labels) * 100
                    resultant_accuracy += accuracy/10.00

            results[opt_i, lr_i] += resultant_accuracy/no_of_experiments

plt.title("Optimizer Comparison")
plt.xlabel("Logarithmically Spaced Learning Rates")
plt.ylabel("Accuracy % (avg. of last 10 epochs)")
for i, optimizer in enumerate(optimizers):
    plt.plot(lr_rates, results[i], label=optimizer)
    plt.legend()
plt.xscale("log")
plt.show()
