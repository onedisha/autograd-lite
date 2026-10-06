import numpy as np


def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing numpy broadcasting."""
    # 1. sum away leading dims that broadcasting prepended
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2. sum over dims that were size 1 in the original but got stretched
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    # makes `ndarray <op> Tensor` fall back to Tensor's reflected methods
    __array_ufunc__ = None

    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, data={self.data}, grad={self.grad})"

    # ---------- elementwise binary ops (these can broadcast) ----------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    # ---------- matrix multiply ----------
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.data.ndim == 2 and other.data.ndim == 2, \
            f"matmul needs 2-D tensors, got {self.data.shape} @ {other.data.shape}"
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        return Tensor(other) @ self

    # ---------- reductions ----------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)   # put the summed axis back
            self.grad += np.broadcast_to(g, self.data.shape)
        out._backward = _backward
        return out

    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    # ---------- unary ops ----------
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # ---------- backprop ----------
    def backward(self):
        # iterative topological sort (no recursion-limit problems on deep graphs)
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._prev:
                stack.append((child, False))

        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()

    # ---------- convenience ops built from the above ----------
    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self ** -1