import math

class Value:
    def __init__(self, data, _children=(), _op='') :
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
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

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f'**{k}')
        def _backward():
            self.grad += k * self.data**(k-1) * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1-t **2) * out.grad
        out._backward = _backward
        return out
    def __neg__(self): return self * -1                   
    def __sub__(self, other): return self + (-other)      
    def __truediv__(self, other): return self * other ** -1    
    def __radd__(self, other): return self + other         
    def __rmul__(self, other): return self * other         

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
        for v in reversed(topo):              
            v._backward()                     

if __name__ == '__main__':
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(7.0)
    x = a * b + c                             
    y = x.tanh()                              
    L = y * a + b ** 2 / c                    
    L.backward()
    print('내 엔진  :', L.data, a.grad, b.grad, c.grad)

    import torch                              
    ta = torch.tensor(2.0, dtype=torch.double, requires_grad=True)    
    tb = torch.tensor(-3.0, dtype=torch.double, requires_grad=True)   
    tc = torch.tensor(7.0, dtype=torch.double, requires_grad=True)
    tL = torch.tanh(ta * tb + tc) * ta + tb ** 2 / tc
    tL.backward()
    print('PyTorch  :', tL.item(), ta.grad.item(), tb.grad.item(), tc.grad.item())
    