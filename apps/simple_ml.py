"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_cifar10(model, train_dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
                  lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss(), 
                  device=ndl.cpu(), val_dataloader=None):
    """
    Performs {n_epochs} epochs of training on CIFAR-10.

    Args:
        model: nn.Module instance
        train_dataloader: Training dataloader instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: loss function instance
        device: computation device
        val_dataloader: Validation dataloader for evaluation during training

    Returns:
        train_history: dictionary containing training history
    """
    np.random.seed(4)
    model.train()
    
    # Initialize optimizer
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_history = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }
    
    for epoch in range(n_epochs):
        model.train()
        correct, total_loss, total_samples = 0, 0, 0
        
        # Training phase
        for batch_idx, batch in enumerate(train_dataloader):
            opt.reset_grad()
            X, y = batch
            X = ndl.Tensor(X, device=device)
            y = ndl.Tensor(y, device=device)
            
            # Forward pass
            out = model(X)
            
            # Calculate accuracy
            preds = np.argmax(out.numpy(), axis=1)
            correct += np.sum(preds == y.numpy())
            total_samples += y.shape[0]
            
            # Calculate loss
            loss = loss_fn(out, y)
            total_loss += loss.data.numpy().item() * y.shape[0]
            
            # Backward pass
            loss.backward()
            opt.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{n_epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.data.numpy().item():.4f}')
        
        # Calculate training metrics
        train_acc = correct / total_samples
        train_loss = total_loss / total_samples
        
        train_history['train_acc'].append(train_acc)
        train_history['train_loss'].append(train_loss)
        
        # Validation phase
        if val_dataloader is not None:
            val_acc, val_loss = evaluate_cifar10(model, val_dataloader, loss_fn, device)
            train_history['val_acc'].append(val_acc)
            train_history['val_loss'].append(val_loss)
            
            print(f'Epoch {epoch+1}/{n_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        else:
            print(f'Epoch {epoch+1}/{n_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    
    return train_history


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss(), device=ndl.cpu()):
    """
    Computes the test accuracy and loss of the model on CIFAR-10.

    Args:
        model: nn.Module instance
        dataloader: Dataloader instance
        loss_fn: loss function instance
        device: computation device

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    model.eval()
    
    correct, total_loss, total_samples = 0, 0, 0
    
    for batch in dataloader:
        X, y = batch
        X = ndl.Tensor(X, device=device)
        y = ndl.Tensor(y, device=device)
        
        # Forward pass (no gradients needed for evaluation)
        with ndl.no_grad():
            out = model(X)
        
        # Calculate accuracy
        preds = np.argmax(out.numpy(), axis=1)
        correct += np.sum(preds == y.numpy())
        total_samples += y.shape[0]
        
        # Calculate loss
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy().item() * y.shape[0]
    
    avg_acc = correct / total_samples
    avg_loss = total_loss / total_samples
    
    return avg_acc, avg_loss


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    nbatch,batch_size = data.shape
    if opt is None:
        model.eval()
    else:
        model.train()
    correct, total_loss = 0,0.0
    total_samples = 0
    hidden = None
    for i in range(0,nbatch,seq_len):
        # start_idx = i * seq_len
        # if start_idx + seq_len + 1 > nbatch:  # +1 是因为target要往后一个位置
        #     break
        batch,labels = ndl.data.get_batch(data,i,seq_len,device=device,dtype=dtype)
        preds,hidden = model(batch,hidden)
        # detach hidden state
        if isinstance(hidden, tuple):
            h, c = hidden
            hidden = (h.detach(), c.detach())
        else:
            hidden = hidden.detach()
        # Calculate loss
        loss = loss_fn(preds, labels)
        batch_size = labels.shape[0]
        total_loss += loss.data.numpy() * batch_size 
        preds = np.argmax(preds.numpy(),axis=1)
        correct += np.sum(preds == labels.numpy())
        total_samples += batch_size       

        # Backward pass
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            if clip is not None:
                opt.clip_grad_norm(clip)
            opt.step()
    return correct / total_samples, total_loss / total_samples        
    
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn = loss_fn()
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    # train_history = {
    #     'train_acc': [],
    #     'train_loss': [],
    # }
    for i in range(n_epochs):
        acc,loss = epoch_general_ptb(data,model,seq_len,loss_fn,optimizer,clip=clip,device=device,dtype=dtype)
        print(f"epoch {i+1}: loss={loss}, acc={acc}\n")
    return acc,loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn = loss_fn()
    # optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    # valid_history = {
    #     'train_acc': [],
    #     'train_loss': [],
    # }
    acc,loss = epoch_general_ptb(data,model,seq_len,loss_fn,device=device,dtype=dtype)
    return acc,loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
