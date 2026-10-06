# autograd-lite

A tiny automatic differentiation engine and neural network library built on **NumPy tensors**, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

micrograd works on single numbers: every weight and every addition is its own node in the graph. autograd-lite keeps the same core idea (a dynamically built graph and reverse-mode backpropagation) but each node holds a whole NumPy array. That is much closer to how real frameworks like PyTorch work, and it means a layer is one matrix multiply instead of a Python loop over neurons.

## Features

- **`Tensor`** with reverse-mode autodiff (`backward()`)
- Operators: `+`, `-`, `*`, `/`, `**` (int/float powers), `@` (matmul)
- Functions: `sum` (with `axis` / `keepdims`), `mean`, `relu`, `tanh`, `exp`, `log`
- **Broadcasting-aware gradients**: gradients are summed back to each operand's original shape
- Works with NumPy on either side of an operator (`ndarray @ Tensor`, `ndarray - Tensor`, ...)
- **`Layer` / `MLP`** where each layer stores one weight matrix `W` of shape `(nin, nout)` and a bias `b` of shape `(nout,)`, with He initialization
- **`draw_dot`**, a Graphviz visualization of the computation graph (shape, data and grad for every tensor)

## Project structure

```
autograd-lite/
├── autogradlite/
│   ├── __init__.py
│   ├── engine.py      # Tensor and backpropagation
│   ├── nn.py          # Module, Layer, MLP
│   └── viz.py         # draw_dot: computation graph visualization
├── demo.ipynb         # walkthrough, gradient checks and training
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/onedisha/autograd-lite.git
cd autograd-lite

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

**Optional: Graphviz.** The graph drawings need the Graphviz program itself (the `dot` executable), which `pip` cannot install. Get it from [graphviz.org/download](https://graphviz.org/download/) and make sure `dot` is on your PATH. Without it, everything else still works and the notebook just skips the drawings.

## Quick start

### Autodiff

```python
from autogradlite.engine import Tensor

a = Tensor([[1., 2.], [3., 4.]])
w = Tensor([[0.5], [-1.0]])

loss = ((a @ w) ** 2).mean()
loss.backward()

print(loss.data)     # 4.25
print(w.grad)        # [[-9.], [-13.]]
```

### Training a network

Inputs are 2-D arrays of shape `(num_samples, num_features)`, one row per sample. Targets must have the same shape as the predictions.

```python
import numpy as np
from autogradlite.nn import MLP

X = np.array([[2, 3, -1], [3, -1, .5], [.5, 1, 1], [1, 1, -1]], dtype=np.float32)
y = np.array([[1], [-1], [-1], [1]], dtype=np.float32)   # shape (4, 1), like the predictions

model = MLP(3, [4, 4, 1])      # 3 inputs -> 4 -> 4 -> 1 (ReLU hidden layers, linear output)

for step in range(200):
    loss = ((model(X) - y) ** 2).mean()   # forward + MSE loss
    model.zero_grad()                     # clear old gradients
    loss.backward()                       # backpropagate
    for p in model.parameters():          # gradient descent step
        p.data -= 0.05 * p.grad

print(model(X).data.round(2))   # close to the targets
```

A single sample must also be 2-D, e.g. `X[0:1]` with shape `(1, 3)`.

### Visualizing the graph

```python
from autogradlite.viz import draw_dot

out = model(X[:1])
out.backward()
draw_dot(out)    # renders in a notebook; needs Graphviz installed
```

## Demo notebook

`demo.ipynb` runs the whole library end to end:

1. A scalar graph with hand-checked gradients
2. Every operation gradient-checked against finite differences, plus broadcasting and NumPy interop checks
3. The MLP's structure, parameter shapes and computation graph
4. A gradient check on every parameter of the network
5. Training on a single sample and on a small batch
6. Final dtype and shape checks

Open it in VS Code or run `jupyter notebook demo.ipynb`.

## Things to know

- **Tensors are `float32`.** Everything is cast to float32 when a `Tensor` is created, as in most deep learning libraries.
- **`matmul` needs 2-D tensors.** Higher-dimensional and 1-D matmul are not supported yet.
- **Watch target shapes.** If predictions are `(batch, 1)` and targets are `(batch,)`, NumPy broadcasts `pred - y` into a `(batch, batch)` matrix and the loss is silently wrong. Reshape targets to match, e.g. `y.reshape(-1, 1)`.
- **Small ReLU networks can have dead units.** Occasionally a tiny network starts with all hidden units inactive and stops learning. Try a different random seed.

## Acknowledgements

Based on the ideas in [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy, and his video [The spelled-out intro to neural networks and back-propagation](https://www.youtube.com/watch?v=VMj-3S1tku0).

## License

MIT. See [LICENSE](LICENSE).