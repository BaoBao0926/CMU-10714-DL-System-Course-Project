import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    block = nn.nn_basic.Sequential(
      nn.nn_basic.Linear(in_features=dim, out_features=hidden_dim),
      norm(hidden_dim),
      nn.nn_basic.ReLU(),
      nn.nn_basic.Dropout(drop_prob),
      nn.nn_basic.Linear(in_features=hidden_dim, out_features=dim),
      norm(dim)
    )
    aa = nn.nn_basic.Sequential(
      nn.nn_basic.Residual(block),
      nn.nn_basic.ReLU()
    )
    return aa


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    return nn.nn_basic.Sequential(
      nn.nn_basic.Linear(dim, hidden_dim),
      nn.nn_basic.ReLU(),
      *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
      nn.nn_basic.Linear(hidden_dim, num_classes)
    )

def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    if opt is not None:
        model.train()
    else:
        model.eval()

    total_loss, total_err, n = 0.0, 0.0, 0
    loss_fn = nn.SoftmaxLoss()

    for X, y in dataloader:
        # forward propogation
        X = X.reshape((X.shape[0], -1)) 
        out = model(X)
        loss = loss_fn(out, y)

        # backward propogation
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()

        # get losst
        total_loss += float(loss.numpy()) * X.shape[0]
        # get error rate
        preds = np.argmax(out.numpy(), axis=1) 
        total_err += np.sum(preds != y.numpy())  
        n += X.shape[0]

    return float(total_err / n), total_loss / n


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    # get dataset
    train_dataset = ndl.data.MNISTDataset(data_dir + "/train-images-idx3-ubyte.gz",
                        data_dir + "/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(data_dir + "/t10k-images-idx3-ubyte.gz",
                        data_dir + "/t10k-labels-idx1-ubyte.gz")
    # get dataloader
    train_loader = ndl.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_dataset, batch_size)

    # get model
    model = MLPResNet(dim=28*28, hidden_dim=hidden_dim, num_classes=10)

    # get optimizer
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training process
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt)
        test_err, test_loss = epoch(test_loader, model)

    return train_err, train_loss, test_err, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
