import numpy as np
import torch
import matplotlib.pyplot as plt


def create_data_set():
    x = np.arange(100, dtype=np.float32)
    y = .4 * x + 3
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return x, y


class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    input_dim = 1
    output_dim = 1
    learning_rate = 0.0001
    epochs = 10000

    x, y = create_data_set()

    model = LinearRegression(input_dim, output_dim)

    x_train = torch.from_numpy(x)
    y_train = torch.from_numpy(y)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # forward pass
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        if epoch % 20 == 0:
            print(epoch, loss.item())
        # backward pass + optimization
        optimizer.zero_grad()
        # get gradients w.r.t to parameters
        loss.backward()
        # update parameters
        optimizer.step()

    # Plot the graph
    predicted = model(torch.from_numpy(x)).detach().numpy()
    plt.plot(x, y, 'ro', label='Original data')
    plt.plot(x, predicted, label='Fitted line')
    plt.legend()
    plt.show()
