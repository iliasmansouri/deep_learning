import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


def create_data_set():
    x, y = make_classification(
        n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1
    )
    return x, y


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.sigmoid(self.linear(x))


if __name__ == "__main__":
    input_dim = 2
    output_dim = 1
    learning_rate = 0.001
    epochs = 10000

    x, y = create_data_set()

    model = LogisticRegression(input_dim, output_dim)

    x_train = Variable(torch.Tensor(x))
    y_train = Variable(torch.Tensor(y))

    loss_fn = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # forward pass
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        if epoch % 1000 == 0:
            print(epoch, loss.item())
        # backward pass + optimization
        optimizer.zero_grad()
        # get gradients w.r.t to parameters
        loss.backward()
        # update parameters
        optimizer.step()

    w = list(model.parameters())
    w0 = w[0].data.numpy()
    w1 = w[1].data.numpy()

    # plot data and separating line
    plt.scatter(x[:, 0], x[:, 1], c=y, s=25, alpha=0.5)
    x_axis = np.linspace(-6, 6, 100)
    y_axis = -(w1[0] + x_axis * w0[0][0]) / w0[0][1]
    (line_up,) = plt.plot(x_axis, y_axis, "r--", label="Separation Line")
    plt.legend(handles=[line_up])
    plt.xlabel("X(1)")
    plt.ylabel("X(2)")
    plt.show()
