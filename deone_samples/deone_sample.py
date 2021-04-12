import sys

print("------start--------")
sys.path.append("C:/Users/cyden/PycharmProjects/")
sys.path.append("D:/")
sys.path.append("/Users/leekangkern/PycharmProjects/")
print(sys.path)

import numpy as np
import deone
import deone.datasets
import deone.utils as utils
from deone import Function
from deone import Variable
from deone import Model, MLP
from deone import optimizers

import deone.functions as F
import deone.layers as L
from deone.layers import Layer

#from deone.utils import plot_dot_graph
import math
import matplotlib.pyplot as plt


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z

def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

def f(x):
    y = x**4 - 2*x**2
    return y

"""
" y = ax + b 형태의 점 무리들을 생성하는 코드 
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1,1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y
"""


def mean_squared_error_simple(x0, x1):
    diff = x0 - x1
    return F.sum(diff**2) / len(diff)

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

def predict(model, x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y


np.random.seed(0)

"""
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000
hidden_size = 10

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y
"""
"""
#model = TwoLayerNet(hidden_size, 1)
model = MLP((hidden_size, 1))
#optimizer = optimizers.SGD(lr)
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    #for p in model.params():
    #    p.data -= lr * p.grad.data
    optimizer.update()

    if i % 1000 == 0 :
        print(loss)

#model.plot(x)

x = Variable(np.array([[1,2,3],[4,5,6]]))
ind = np.array([0,0,1])
y = F.get_item(x, ind)
print(y)

model = MLP((10, 3))
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, 3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
#p = F.softmax_simple(y)
loss = F.softmax_cross_entropy_simple(y, t)
print(loss)
"""

x = np.random.rand(1, 3, 7, 7)
#print(x.shape)
#print(x)

col1 = F.im2col(x, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)
print(col1)
quit()

from deone.datasets import Spiral
from deone.dataloaders import DataLoader

x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with deone.core.test_mode():
    y = F.dropout(x)
    print(y)

quit()


train_set = deone.datasets.MNIST(train=True, transform=None)
test_set = deone.datasets.MNIST(train=False, transform=None)

print(train_set)
print(test_set)

#quit()
max_epoch = 300
batch_size = 10
hidden_size = 10
lr = 1.0

#train_set = deone.datasets.Spiral()
train_set = Spiral(train=True)
test_set = Spiral(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch : {}'.format(epoch+1))
    print('train loss : {:.4f}, accuracy : {:.4f}'.format(sum_loss / len(train_set),
                                                          sum_acc / len(train_set)))

    sum_loss, sum_acc = 0,0
    with deone.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    print('test loss : {:.4f}, accuracy : {:.4f}'.format(sum_loss / len(train_set),
                                                          sum_acc / len(train_set)))
