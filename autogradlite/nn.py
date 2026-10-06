import numpy as np
from autogradlite.engine import Tensor


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []


class Layer(Module):
    """Fully connected layer: out = x @ W + b  (optionally followed by ReLU).

    x: (batch, nin)   W: (nin, nout)   b: (nout,)   out: (batch, nout)
    Column j of W holds the weights of neuron j.
    """

    def __init__(self, nin, nout, nonlin=True):
        self.W = Tensor(np.random.randn(nin, nout) * np.sqrt(2.0 / nin))  # He init
        self.b = Tensor(np.zeros(nout))
        self.nonlin = nonlin

    def __call__(self, x):
        out = x @ self.W + self.b       # (batch,nin)@(nin,nout) + (nout,) broadcast
        return out.relu() if self.nonlin else out

    def parameters(self):
        return [self.W, self.b]


class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        # last layer is linear, all others use ReLU
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
                       for i in range(len(nouts))]

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]