import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

class ConvBNRelu(ndl.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.conv = nn.Conv(in_channels, out_channels, kernel_size, stride, bias, device, dtype)
        self.bn2d = nn.BatchNorm2d(dim = out_channels,device = device,dtype = dtype)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.bn2d(x)
        return self.relu(x)



class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.device = device
        self.dtype = dtype
        middle_blocks_params = [(32,32,3,1),(32,64,3,2),(64,128,3,2),(128,128,3,1)]
        self.input_blocks = nn.Sequential(
            ConvBNRelu(3,16,7,4,bias=True,device=device,dtype=dtype),
            ConvBNRelu(16,32,3,2,bias=True,device=device,dtype=dtype)
        )
        self.middle_blocks1 = nn.Sequential(
            ConvBNRelu(*middle_blocks_params[0],bias=True,device=device,dtype=dtype),
            ConvBNRelu(*middle_blocks_params[0],bias=True,device=device,dtype=dtype),
        )
        self.middle_blocks2 = nn.Sequential(
            ConvBNRelu(*middle_blocks_params[1],bias=True,device=device,dtype=dtype),
            ConvBNRelu(*middle_blocks_params[2],bias=True,device=device,dtype=dtype)
        )
        self.middle_blocks3 = nn.Sequential(
            ConvBNRelu(*middle_blocks_params[3],bias=True,device=device,dtype=dtype),
            ConvBNRelu(*middle_blocks_params[3],bias=True,device=device,dtype=dtype)
        )

        #     ConvBNRelu(*middle_blocks_params[1],bias=True,device=device,dtype=dtype),
        #     ConvBNRelu(*middle_blocks_params[2],bias=True,device=device,dtype=dtype),
        #     ConvBNRelu(*middle_blocks_params[3],bias=True,device=device,dtype=dtype),
        #     ConvBNRelu(*middle_blocks_params[3],bias=True,device=device,dtype=dtype)
        # ])
        self.output_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,128,bias=True,device=device,dtype=dtype),
            nn.ReLU(),
            nn.Linear(128,10,bias=True,device=device,dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.input_blocks(x)
        x = nn.Residual(self.middle_blocks1)(x)
        x = self.middle_blocks2(x)
        x = nn.Residual(self.middle_blocks3)(x)
        x = self.output_linear(x)
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size,embedding_size,device=device,dtype=dtype)
        if seq_model=='rnn':
            self.seq_model = nn.RNN(embedding_size,hidden_size,num_layers,bias=True,device=device,dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size,hidden_size,num_layers,bias=True,device=device,dtype=dtype)
        self.linear = nn.Linear(hidden_size,output_size,bias=True,device=device,dtype=dtype)
        self.seq_len =seq_len
        self.num_layers = num_layers
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        x = self.embedding(x) # transfer to embeddings
        # if isinstance(self.seq_model,nn.RNN):
        #     if h is None:
        #         h = nn.init.zeros(self.num_layers,bs,self.hidden_size,device=x.device,dtype=x.dtype)
        #     x,h = self.seq_model(x,h)
        # elif isinstance(self.seq_model,nn.LSTM):
        #     if h is None:
        #         h = (
        #             nn.init.zeros(self.num_layers,bs,self.hidden_size,device=x.device,dtype=x.dtype),
        #             nn.init.zeros(self.num_layers,bs,self.hidden_size,device=x.device,dtype=x.dtype)
        #         )
        x,h = self.seq_model(x,h)
        *_,hidden_size = x.shape
        x = x.reshape((-1,hidden_size))
        x = self.linear(x)
        return x,h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
