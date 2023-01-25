"""
@Author: Zhou Zhiwei
@Date: 2023/1/24 20:29
@Description: neural network components
"""
import random
from engine import Value


class Module:
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


if __name__ == '__main__':
    xs = [
        [2., 3., -1.],
        [3., -1., 0.5],
        [0.5, 1., 1.],
        [1., 1., -1.]
    ]
    ys = [1., -1., -1., 1.]

    neural_net = MLP(3, [4, 4, 1])

    for k in range(100):
        ypred = [neural_net(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for yout, ygt in zip(ypred, ys))

        neural_net.zero_grad()
        loss.backward()

        lr = 1.0 - 0.9 * k / 100
        for p in neural_net.parameters():
            p.data += -0.01 * p.grad
        print(k, loss.data, [y.data for y in ypred])
