"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return(1+ops.exp(-x))**-1
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        assert nonlinearity == 'tanh' or nonlinearity == 'relu' 
        self.nonlinearity = ops.tanh if nonlinearity=='tanh' else ops.relu
        wih_shape = (input_size,hidden_size)
        whh_shape = (hidden_size,hidden_size)
        k = np.sqrt(1/hidden_size)
        self.W_ih = Parameter(init.rand(*wih_shape,low=-k,high=k,requires_grad=True,device=device,dtype=dtype))
        self.W_hh = Parameter(init.rand(*whh_shape,low=-k,high=k,requires_grad=True,device=device,dtype=dtype))
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size,low=-k,high=k,requires_grad=True,device=device,dtype=dtype))
            self.bias_hh = Parameter(init.rand(hidden_size,low=-k,high=k,requires_grad=True,device=device,dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is not None:
            h = X @ self.W_ih + h @ self.W_hh
        else:
            h = X @ self.W_ih
        if (self.bias_ih is not None):
            h = h + ops.broadcast_to(self.bias_ih,h.shape)
        if (self.bias_hh is not None):
            h = h + ops.broadcast_to(self.bias_hh,h.shape)
        h = self.nonlinearity(h)
        return h
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = [RNNCell(input_size,hidden_size,bias,nonlinearity,device,dtype)]
        for i in range(num_layers-1):
            self.rnn_cells.append(RNNCell(hidden_size,hidden_size,bias,nonlinearity,device,dtype))
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        hn = h0
        seq_len,bs,_ = X.shape
        if h0 is None:
            hn = Tensor(np.zeros((self.num_layers,bs,self.hidden_size)),device=X.device,dtype=X.dtype)
        hn = list(ops.split(hn,axis=0))
        X = list(ops.split(X,axis=0))
        for i in range(seq_len):
            for j in range(self.num_layers):
                X[i] = self.rnn_cells[j](X[i],hn[j]) # input list updates to this layer's hidden state h_t^l
                hn[j] = X[i]
        return ops.stack(X,axis=0),ops.stack(hn,axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        wih_shape = (input_size,4*hidden_size)
        whh_shape = (hidden_size,4*hidden_size)
        k = np.sqrt(1/hidden_size)
        self.W_ih = Parameter(init.rand(*wih_shape,low=-k,high=k,requires_grad=True,device=device,dtype=dtype))
        self.W_hh = Parameter(init.rand(*whh_shape,low=-k,high=k,requires_grad=True,device=device,dtype=dtype))
        self.bias_ih = None
        self.bias_hh = None
        self.hidden_size = hidden_size
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size,low=-k,high=k,requires_grad=True,device=device,dtype=dtype))
            self.bias_hh = Parameter(init.rand(4*hidden_size,low=-k,high=k,requires_grad=True,device=device,dtype=dtype))
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs,_ = X.shape
        if h is None:
            h = (# initialize h0 and c0
                Tensor(np.zeros((bs,self.hidden_size)),device=X.device,dtype=X.dtype), 
                Tensor(np.zeros((bs,self.hidden_size)),device=X.device,dtype=X.dtype)
            )
        hn,cn = h
        gates_output = X @ self.W_ih + hn @ self.W_hh
        if (self.bias_ih is not None) and (self.bias_hh is not None):
            gates_output += self.bias_ih.broadcast_to(gates_output.shape)+self.bias_hh.broadcast_to(gates_output.shape)
        gates_output = list(ops.split(gates_output,axis=1))
        i = self.sigmoid(ops.stack(gates_output[0:self.hidden_size],axis=1))
        f = self.sigmoid(ops.stack(gates_output[self.hidden_size:2*self.hidden_size],axis=1))
        g = ops.tanh(ops.stack(gates_output[2*self.hidden_size:3*self.hidden_size],axis=1))
        o = self.sigmoid(ops.stack(gates_output[3*self.hidden_size:4*self.hidden_size],axis=1))
        #i,f,g,o = self.sigmoid(i), self.sigmoid(f), ops.tanh(g), self.sigmoid(o)
        cn = f * cn + i * g
        hn = o * ops.tanh(cn)
        return hn,cn     
        
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.lstm_cells = [LSTMCell(input_size,hidden_size,bias,device,dtype)]
        for i in range(num_layers-1):
            self.lstm_cells.append(LSTMCell(hidden_size,hidden_size,bias,device,dtype))
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len,bs,_ = X.shape
        if h is None:
            hn = Tensor(np.zeros((self.num_layers,bs,self.hidden_size)),device=X.device,dtype=X.dtype)
            #h = hn
            cn = Tensor(np.zeros((self.num_layers,bs,self.hidden_size)),device=X.device,dtype=X.dtype)
            h = (hn,cn)
        hn,cn = h
        X = list(ops.split(X,axis=0))
        hn = list(ops.split(hn,axis=0))
        cn = list(ops.split(cn,axis=0))
        for i in range(seq_len):
            for j in range(self.num_layers):
                X[i],cn[j] = self.lstm_cells[j](X[i],(hn[j],cn[j])) # input list updates to this layer's hidden state h_t^l
                hn[j] = X[i]
        return ops.stack(X,axis=0),(ops.stack(hn,axis=0),ops.stack(cn,axis=0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        weight = init.randn(num_embeddings,embedding_dim,device=device,dtype=dtype,requires_grad=True)
        self.weight = Parameter(weight)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        x_shape =x.shape
        x = init.one_hot(self.num_embeddings,x,device=x.device,dtype=x.dtype) # transform x to one-hot vector
        x = x.reshape((-1,self.num_embeddings)) 
        x = x @ self.weight
        x = x.reshape((*x_shape,self.embedding_dim))
        return x
        ### END YOUR SOLUTION