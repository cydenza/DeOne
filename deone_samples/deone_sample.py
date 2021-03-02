import numpy as np

import sys

print("------start--------")
sys.path.append("C:/Users/cyden/PycharmProjects/")
sys.path.append("D:/")
print(sys.path)

from deone.core_simple import Variable

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

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
#z = sphere(x, y)
#z = matyas(x, y)
z = goldstein(x, y)
z.backward()
print(x.grad, y.grad)

""":arg

"""
from deone.utils import get_dot_graph

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0 + x1

x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'

txt = get_dot_graph(y, verbose=False)
print(txt)

with open('sample.doti', 'w') as o:
    o.write(txt)
