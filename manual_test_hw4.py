import numpy as np
import torch
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
import sys
sys.path.append('./python')
sys.path.append('./apps')
from models import LanguageModel
from simple_ml import *
def backward_check(f, *args, **kwargs):
    eps = 1e-5
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(ndl.Tensor(c, device=args[0].device), out)
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 4.2e-1
    return [g.numpy() for g in backward_grad]

def test_stack_backward(shape, axis, l, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
    A = [ndl.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
    A_t = [torch.Tensor(_A[i]) for i in range(l)]
    for i in range(l):
        A_t[i].requires_grad = True
    ndl.stack(A, axis=axis).sum().backward()
    torch.stack(A_t, dim=axis).sum().backward()
    for i in range(l):
        np.testing.assert_allclose(A_t[i].grad.numpy(), A[i].grad.numpy(), atol=1e-5, rtol=1e-5)

def test_cifar10_dataset(train):
    dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=train)
    if train:
        assert len(dataset) == 50000
    else:
        assert len(dataset) == 10000
    example = dataset[np.random.randint(len(dataset))]
    assert(isinstance(example, tuple))
    X, y = example
    assert isinstance(X, np.ndarray)
    assert X.shape == (3, 32, 32)


def test_broadcast_to(shape, shape_to, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.broadcast_to(_A, shape_to), ndl.broadcast_to(A, shape_to).numpy(), atol=1e-5, rtol=1e-5)

def DoDilate(shape, axes, dilation, backward=False, device=ndl.cpu()):
        X = Rand(*shape, device=device)
        X.requires_grad = True
        Y = ndl.dilate(X, dilation=dilation, axes=axes)
        if backward:
            V = Rand(*Y.shape, device=device, entropy=2)
            Z = (V*Y).sum()
            Z.backward()
            return X.grad
        else:
            return Y
def test_nn_conv_forward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = ndl.nn.Conv(cin, cout, k, stride=stride, device=device)
    x = ndl.init.rand(10, cin, s, s, device=device)

    g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
    g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy())
    result_f = f(x).cached_data.numpy()
    result_g = g(z).data.numpy()
    assert np.linalg.norm(result_f-result_g) < 1e-3

def test_nn_conv_backward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = ndl.nn.Conv(cin, cout, k, stride=stride, device=device)
    x = ndl.init.rand(1, cin, s, s, device=device, requires_grad=True)

    g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
    g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy(), requires_grad=True)
    z.requires_grad = True

    res1 = f(x)
    y1 = res1.sum()

    y2 = g(z).sum()

    y1.backward()
    y2.backward()

    assert np.linalg.norm(g.weight.grad.data.numpy() - f.weight.grad.cached_data.numpy().transpose(3, 2, 0, 1)) < 1e-3, "weight gradients match"
    assert np.linalg.norm(g.bias.grad.data.numpy() - f.bias.grad.cached_data.numpy()) < 1e-3, "bias gradients match"
    assert np.linalg.norm(z.grad.data.numpy() - x.grad.cached_data.numpy()) < 1e-3, "input gradients match"

def test_op_conv(Z_shape, W_shape, stride, padding, backward, device):
    np.random.seed(0)
    import torch
    _Z = np.random.randn(*Z_shape)*5
    _Z = _Z.astype(np.float32)
    _W = np.random.randn(*W_shape)*5
    _W = _W.astype(np.float32)
    Z = ndl.Tensor(_Z, device=device)
    W = ndl.Tensor(_W, device=device)
    y = ndl.conv(Z, W, padding=padding, stride=stride)
    y2 = y.sum()
    if backward:
        y2.backward()
    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad=True
    Wtch = torch.Tensor(_W).float()
    Wtch.requires_grad=True
    out = torch.nn.functional.conv2d(Ztch.permute(0, 3, 1, 2), Wtch.permute(3, 2, 0, 1), padding=padding, stride=stride)
    out2 = out.sum()
    if backward:
        out2.backward()
    if backward:
        err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
        err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    if backward:
        assert err1 < 1e-2, "input grads match"
        assert err2 < 1e-2, "weight grads match"
    assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)

def test_init_kaiming_uniform(device):
    _A = np.random.randn(3, 3, 16, 8)
    A = ndl.Tensor(_A, device=device)
    np.random.seed(0)
    A = ndl.init.kaiming_uniform(16*9, 8*9, shape=A.shape)
    assert abs(A.sum().numpy() - -2.5719218) < 1e-4

def test_resnet9(device):
    def num_params(model):
        return np.sum([np.prod(x.shape) for x in model.parameters()])

    from apps.models import ResNet9
    np.random.seed(0)
    model = ResNet9(device=device)

    assert num_params(model) == 431946

    _A = np.random.randn(2, 3, 32, 32)
    A = ndl.Tensor(_A, device=device)
    y = model(A)

    assert np.linalg.norm(np.array([[-1.8912625 ,  0.64833605,  1.9400386 ,  1.1435282 ,  1.89777   ,
         2.9039745 , -0.10433993,  0.35458302, -0.5684191 ,  2.6178317 ],
       [-0.2905612 , -0.4147861 ,  0.90268034,  0.46530387,  1.3335679 ,
         1.8534894 , -0.1867125 , -2.4298222 , -0.5344223 ,  4.362149  ]]) - y.numpy()) < 1e-2

def one_iter_of_cifar10_training(dataloader, model, niter=1, loss_fn=ndl.nn.SoftmaxLoss(), opt=None, device=None):
    np.random.seed(4)
    model.train()
    correct, total_loss = 0, 0
    i = 1
    for batch in dataloader:
        opt.reset_grad()
        X, y = batch
        X,y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
        out = model(X)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy() * y.shape[0]
        loss.backward()
        opt.step()
        if i >= niter:
            break
        i += 1
    return correct/(y.shape[0]*niter), total_loss/(y.shape[0]*niter)

def test_train_cifar10(device):
    import sys
    sys.path.append('.')
    from apps.models import ResNet9
    model = ResNet9(device=device)

    np.random.seed(1)
    dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    dataloader = ndl.data.DataLoader(\
             dataset=dataset,
             batch_size=128,
             shuffle=True
             )
    np.random.seed(1)
    model = ResNet9(device=device, dtype="float32")
    out = one_iter_of_cifar10_training(dataloader, model, niter=2, opt=ndl.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001), device=device)
    assert np.linalg.norm(np.array(list(out), dtype=object) - np.array([0.11328125, 3.7944627])) < 1e-2

def gradient_check(f, *args, tol=1e-6, backward=False, device=None,**kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [
            x.numpy()
            for x in out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape),device=device), out)
        ]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i]) for i in range(len(args))
    )
    assert error < tol
    return computed_grads

def test_matmul_batched_backward(device):
    backward_check(
        ndl.transpose,
        ndl.Tensor(np.random.randn(4, 5,6),device=device),
        axes = (0,2)
    )
    backward_check(
        ndl.transpose,
        ndl.Tensor(np.random.randn(4, 5,6),device=device),
        axes = None
    )
    backward_check(
        ndl.relu,
        ndl.Tensor(np.random.randn(6,4,2),device=device),
    )

def test_rnn_cell(batch_size, input_size, hidden_size, bias, init_hidden, nonlinearity, device):
    np.random.seed(1)
    x = np.random.randn(batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.RNNCell(input_size, hidden_size, nonlinearity=nonlinearity, bias=bias)
    if init_hidden:
        h_ = model_(torch.tensor(x), torch.tensor(h0))
    else:
        h_ = model_(torch.tensor(x), None)

    model = nn.RNNCell(input_size, hidden_size, device=device, bias=bias, nonlinearity=nonlinearity)
    model.W_ih = ndl.Tensor(model_.weight_ih.detach().numpy().transpose(), device=device)
    model.W_hh = ndl.Tensor(model_.weight_hh.detach().numpy().transpose(), device=device)
    if bias:
        model.bias_ih = ndl.Tensor(model_.bias_ih.detach().numpy(), device=device)
        model.bias_hh = ndl.Tensor(model_.bias_hh.detach().numpy(), device=device)
    if init_hidden:
        h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))
    else:
        h = model(ndl.Tensor(x, device=device), None)
    assert h.device == device
    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    h.sum().backward()
    h_.sum().backward()
    np.testing.assert_allclose(model_.weight_ih.grad.detach().numpy().transpose(), model.W_ih.grad.numpy(), atol=1e-5, rtol=1e-5)

def test_rnn(seq_length, num_layers, batch_size, input_size, hidden_size, bias, init_hidden, nonlinearity, device):
    x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers, bias=bias, nonlinearity=nonlinearity)
    if init_hidden:
        output_, h_ = model_(torch.tensor(x), torch.tensor(h0))
    else:
        output_, h_ = model_(torch.tensor(x), None)

    model = nn.RNN(input_size, hidden_size, num_layers, bias, device=device, nonlinearity=nonlinearity)
    for k in range(num_layers):
        model.rnn_cells[k].W_ih = ndl.Tensor(getattr(model_, f'weight_ih_l{k}').detach().numpy().transpose(), device=device)
        model.rnn_cells[k].W_hh = ndl.Tensor(getattr(model_, f'weight_hh_l{k}').detach().numpy().transpose(), device=device)
        if bias:
            model.rnn_cells[k].bias_ih = ndl.Tensor(getattr(model_, f'bias_ih_l{k}').detach().numpy(), device=device)
            model.rnn_cells[k].bias_hh = ndl.Tensor(getattr(model_, f'bias_hh_l{k}').detach().numpy(), device=device)
    if init_hidden:
        output, h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))
    else:
        output, h = model(ndl.Tensor(x, device=device), None)

    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5)

    output.sum().backward()
    output_.sum().backward()
    np.testing.assert_allclose(model.rnn_cells[0].W_ih.grad.detach().numpy(), model_.weight_ih_l0.grad.numpy().transpose(), atol=1e-5, rtol=1e-5)

def test_lstm_cell(batch_size, input_size, hidden_size, bias, init_hidden, device):
    x = np.random.randn(batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
    c0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.LSTMCell(input_size, hidden_size, bias=bias)
    if init_hidden:
        h_, c_ = model_(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
    else:
        h_, c_ = model_(torch.tensor(x), None)

    model = nn.LSTMCell(input_size, hidden_size, device=device, bias=bias)

    model.W_ih = ndl.Tensor(model_.weight_ih.detach().numpy().transpose(), device=device)
    model.W_hh = ndl.Tensor(model_.weight_hh.detach().numpy().transpose(), device=device)
    if bias:
        model.bias_ih = ndl.Tensor(model_.bias_ih.detach().numpy(), device=device)
        model.bias_hh = ndl.Tensor(model_.bias_hh.detach().numpy(), device=device)

    if init_hidden:
        h, c = model(ndl.Tensor(x, device=device), (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)))
    else:
        h, c = model(ndl.Tensor(x, device=device), None)
    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5)

    h.sum().backward()
    h_.sum().backward()
    np.testing.assert_allclose(model_.weight_ih.grad.detach().numpy().transpose(), model.W_ih.grad.numpy(), atol=1e-5, rtol=1e-5)

def test_lstm(seq_length, num_layers, batch_size, input_size, hidden_size, bias, init_hidden, device):
    x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
    c0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.LSTM(input_size, hidden_size, bias=bias, num_layers=num_layers)
    if init_hidden:
        output_, (h_, c_) = model_(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
    else:
        output_, (h_, c_) = model_(torch.tensor(x), None)

    model = nn.LSTM(input_size, hidden_size, num_layers, bias, device=device)
    for k in range(num_layers):
        model.lstm_cells[k].W_ih = ndl.Tensor(getattr(model_, f'weight_ih_l{k}').detach().numpy().transpose(), device=device)
        model.lstm_cells[k].W_hh = ndl.Tensor(getattr(model_, f'weight_hh_l{k}').detach().numpy().transpose(), device=device)
        if bias:
            model.lstm_cells[k].bias_ih = ndl.Tensor(getattr(model_, f'bias_ih_l{k}').detach().numpy(), device=device)
            model.lstm_cells[k].bias_hh = ndl.Tensor(getattr(model_, f'bias_hh_l{k}').detach().numpy(), device=device)
    if init_hidden:
        output, (h, c) = model(ndl.Tensor(x, device=device), (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)))
    else:
        output, (h, c) = model(ndl.Tensor(x, device=device), None)

    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5)

    output.sum().backward()
    output_.sum().backward()
    np.testing.assert_allclose(model.lstm_cells[0].W_ih.grad.detach().numpy(), model_.weight_ih_l0.grad.numpy().transpose(), atol=1e-5, rtol=1e-5)
def test_ptb_dataset(batch_size, bptt, train, device):
    # TODO update with more tests?
    corpus = ndl.data.Corpus("data/ptb")
    if train:
        data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    else:
        data = ndl.data.batchify(corpus.test, batch_size, device=device, dtype="float32")
    X, y = ndl.data.get_batch(data, np.random.randint(len(data)), bptt, device=device)
    assert X.shape == (bptt, batch_size)
    assert y.shape == (bptt * batch_size,)
    assert isinstance(X, ndl.Tensor)
    assert X.dtype == 'float32'
    assert X.device == device
    assert isinstance(X.cached_data, nd.NDArray)
    ntokens = len(corpus.dictionary)
    assert ntokens == 10000

def test_language_model_implementation(seq_length, num_layers, batch_size, embedding_size, hidden_size,
                        init_hidden, output_size, seq_model, device):
    #TODO add test for just nn.embedding?
    import sys
    sys.path.append('./python')
    sys.path.append('./apps')
    from models import LanguageModel
    x = np.random.randint(0, output_size, (seq_length, batch_size)).astype(np.float32)
    h0 = ndl.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)
    c0 = ndl.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)

    model = LanguageModel(embedding_size, output_size, hidden_size, num_layers, seq_model, device=device)
    if init_hidden:
        if seq_model == 'lstm':
            h = (h0, c0)
        elif seq_model == 'rnn':
            h = h0
        output, h_ = model(ndl.Tensor(x, device=device), h)
    else:
        output, h_ = model(ndl.Tensor(x, device=device), None)

    if seq_model == 'lstm':
        assert isinstance(h_, tuple)
        h0_, c0_ = h_
        assert c0_.shape == (num_layers, batch_size, hidden_size)
    elif seq_model == 'rnn':
        h0_ = h_
    assert h0_.shape == (num_layers, batch_size, hidden_size)
    assert output.shape == (batch_size * seq_length, output_size)
    #TODO actually test values
    output.backward()
    for p in model.parameters():
        assert p.grad is not None

def test_language_model_training(device):
    ### 这个测试如果单独跑返回的结果和直接用pytest命令跑的结果不一样，我不知道为什么，如果想验证的话直接用hw4.ipynb的local test验证吧 ###
    corpus = ndl.data.Corpus("data/ptb", max_lines=20)
    seq_len = 10
    num_examples = 100
    batch_size = 16
    seq_model = 'rnn'
    num_layers = 2
    hidden_size = 10
    n_epochs=2
    train_data = ndl.data.batchify(corpus.train, batch_size=batch_size, device=device, dtype="float32")
    model = LanguageModel(30, len(corpus.dictionary), hidden_size=hidden_size, num_layers=num_layers, seq_model=seq_model, device=device)
    train_acc, train_loss = train_ptb(model, train_data, seq_len=seq_len, n_epochs=n_epochs, device=device)
    test_acc, test_loss = evaluate_ptb(model, train_data, seq_len=seq_len, device=device)
    if str(device) == "cpu()":
        np.testing.assert_allclose(5.809671, train_loss, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(5.391172, test_loss, atol=1e-5, rtol=1e-5)
    elif str(device) == "cuda()":
        np.testing.assert_allclose(5.632849, train_loss, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(5.417056, test_loss, atol=1e-5, rtol=1e-5)

device = ndl.cpu()
#test_matmul_batched_backward(ndl.cpu())
#test_train_cifar10(ndl.cpu())
#test_rnn_cell(batch_size=15,input_size=1,hidden_size=12,bias=True,init_hidden=True,nonlinearity='tanh',device=device)
#test_rnn(seq_length=13,num_layers=2,batch_size=15,input_size=1,hidden_size=12,bias=True,init_hidden=True,nonlinearity='tanh',device=device)
#test_lstm_cell(batch_size=15,input_size=1,hidden_size=12,bias=True,init_hidden=True,device=device)
#test_lstm(seq_length=13,num_layers=2,batch_size=15,input_size=1,hidden_size=12,bias=True,init_hidden=True,device=device)
#test_ptb_dataset(batch_size=15,bptt=32,train=True,device=device)
# test_language_model_implementation(seq_length=13,num_layers=2,batch_size=15,embedding_size=34,hidden_size=12,
#                                    init_hidden=False,output_size=1000,seq_model='lstm',device=device)
test_language_model_training(device)


# params = (32, 8, 16, 3, 2)
# test_nn_conv_forward(*params,device=ndl.cpu())
# test_broadcast_to(shape=(1,1,1),shape_to=(4,1,6),device=ndl.cpu())
#test_stack_backward(shape=(5,5),axis=0,l=1,device=device)
#test_cifar10_dataset(True)
#test_init_kaiming_uniform(ndl.cpu())

# DoDilate((2, 2), (2,), 1,backward=True)
# A = np.arange(48).reshape(1,3,4,4)
# weight = np.random.randn(2,2,3,8)
# A_ = nd.NDArray(A)
# A_ = A_.permute((0,2,3,1))
# print(A_)
# weight = nd.NDArray(weight)
# S = 2
# P = 1
# N,H,W,Cin = A_.shape
# K,_,_,Cout = weight.shape
# if P:
#     A_ = A_.pad(axes=((0,0),(P,P),(P,P),(0,0)))
#     N, H_pad, W_pad, Cin = A_.shape
# else:
#     H_pad,W_pad = H,W
# Ns,Hs,Ws,Cs = A_.strides
# H_out = (H_pad - K) // S + 1
# W_out = (W_pad - K) // S + 1
# inner_dim = K * K *Cin
# outer_dim = N * H_out * W_out
# new_shape = (N, H_out,W_out, K, K, Cin)
# new_strides = (Ns,Hs*S,Ws*S,Hs,Ws,Cs)
# A_ = A_.make(shape=new_shape,strides=new_strides,handle=A_._handle, device=A_.device, offset=A_._offset).compact()
# A_ = A_.reshape((outer_dim,inner_dim))
# print(A_)
# params = ( (14, 8, 16, 3, 1) )
# test_nn_conv_backward(*params,ndl.cpu())
# device = ndl.cpu()
# # #device = ndl.cuda()
# # shape = [(2,1,2,3),(8,3,2), (4, 5, 6)]
# # axes=3
# # l =2
# # _A = np.random.randn(*shape[0]).astype(np.float32)
# # A = ndl.Tensor(nd.array(_A), device=device)
# # backward_check(ndl.logsumexp, A, axes=axes)

# # train = True
# # dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=train)
# # if train:
# #     assert len(dataset) == 50000
# # else:
# #     assert len(dataset) == 10000
# # example = dataset[np.random.randint(len(dataset))]
# # assert(isinstance(example, tuple))
# # X, y = example
# # assert isinstance(X, np.ndarray)
# # assert X.shape == (3, 32, 32)
# device = ndl.cpu()
# np.random.seed(0)
# shape, padding = (10, 32, 32, 8),( (0, 0), (2, 2), (2, 2), (0, 0) )
# _A = np.random.randn(*shape)
# _B = np.pad(_A, padding)
# A = nd.NDArray(_A, device=device)
# B = A.pad(padding)

# assert np.linalg.norm(A.numpy() - _A) < 1e-4