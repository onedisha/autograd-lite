import numpy as np
from graphviz import Digraph


def trace(root):
    """Collect every node and edge reachable backwards from `root`."""
    nodes, edges = set(), set()
    stack = [root]                       # iterative DFS: no recursion-limit problems
    while stack:
        v = stack.pop()
        if v in nodes:
            continue
        nodes.add(v)
        for child in v._prev:
            edges.add((child, v))
            stack.append(child)
    return nodes, edges


def _fmt(arr, name, max_values):
    """Format an array for a node label: value if scalar, values if tiny, else its norm."""
    if arr.size == 1:
        return f"{name} {arr.item():.4f}"
    if arr.size <= max_values:
        return f"{name} {np.array2string(arr.ravel(), precision=2)}"
    return f"{name} norm {np.linalg.norm(arr):.3g}"


def draw_dot(root, format='svg', rankdir='LR', max_values=6):
    """Draw the computation graph ending at `root`.

    Each tensor is a record showing its shape, data and grad. Tensors with at most
    `max_values` elements show their values; larger ones show their L2 norm.
    Name a node with `t.label = 'w1'` to have it shown.
    """
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))                                   # unique node id
        name = getattr(n, 'label', '') or ''
        head = f"{name} {tuple(n.data.shape)}".strip()
        fields = [head, _fmt(n.data, 'data', max_values), _fmt(n.grad, 'grad', max_values)]
        dot.node(name=uid, label="{ " + " | ".join(fields) + " }", shape='record')

        # if this tensor was produced by an operation, add a separate op node
        if n._op:
            op_uid = uid + '_' + n._op
            dot.node(name=op_uid, label=n._op, shape='circle',
                     style='filled', fillcolor='lightblue')
            dot.edge(op_uid, uid)                          # op -> result

    for child, parent in edges:
        if parent._op:
            dot.edge(str(id(child)), str(id(parent)) + '_' + parent._op)   # input -> op

    return dot