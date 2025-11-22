"""Optimization module"""
import needle as ndl
import numpy as np
from needle.ops.ops_mathematic import *


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
      for param in self.params:
        if param.grad is None:   # if there is no gradient in one parameter. just skip it
          continue

        grad = param.grad + np.array(self.weight_decay, dtype=param.dtype) * param.data

        # update momentm in self.u (self.u is a dictionary)
        if param not in self.u:
          self.u[param] = (1-self.momentum) * grad
          # self.u[param] = grad
        else:
          self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad
    
        param.data -= self.lr * self.u[param]

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):


      self.t += 1
      for param in self.params:
          if param.grad is None:
              continue

          # get data
          grad = param.grad.data
          param_data = param.data
          # add L2 norm term
          grad = grad + self.weight_decay * param_data

          if param not in self.m:
              self.m[param] = np.zeros_like(param_data)
              self.v[param] = np.zeros_like(param_data)
          # updae m and v
          self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
          self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)

          # bias correction
          m_hat = self.m[param] / (1 - self.beta1 ** self.t)
          v_hat = self.v[param] / (1 - self.beta2 ** self.t)

          update = self.lr * m_hat / (power_scalar(v_hat, 0.5) + self.eps)
          param.data = param_data - update










