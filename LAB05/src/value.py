import numpy as np

class Value:
    def __init__(self, data, _children=(), op=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None
        self.op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):

        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):

        out = Value(
            max(0, self.data),
            (self,),
            'relu'
        )

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build(child)

                topo.append(v)

        build(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()