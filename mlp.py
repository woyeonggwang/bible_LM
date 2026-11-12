import random
from autograd import Value               

random.seed(42)                          

class Neuron:                            
    def __init__(self, nin):             
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]   
        self.b = Value(0.0)              

    def __call__(self, x):               
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
       
        return act.tanh()                

    def parameters(self):                
        return self.w + [self.b]

class Layer:                             
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]   

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]                 
        return outs[0] if len(outs) == 1 else outs          

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]   

class MLP:                               
    def __init__(self, nin, nouts):      
        sz = [nin] + nouts               
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]   

    def __call__(self, x):
        for layer in self.layers:       
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]              

model = MLP(3, [4, 4, 1])
print('파라미터 수:', len(model.parameters()))   

for step in range(1, 101):
    preds = [model(x) for x in xs]                           
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))      
    for p in model.parameters():
        p.grad = 0.0                     
    loss.backward()                      
    for p in model.parameters():
        p.data -= 0.05 * p.grad          
    if step % 10 == 0:
        print(f'step {step}: loss {loss.data:.4f}')

print('예측:', [round(p.data, 3) for p in preds])
print('정답:', ys)