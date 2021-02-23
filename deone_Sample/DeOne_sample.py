import numpy as np

import sys

print("------start--------")
#sys.path.append("C:/Users/cyden/PycharmProjects/deone/")
print(sys.path)

from deone.core_simple import Variable

def sphere(x, y):
    z = x ** 2 + y ** 2
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y)
z.backward()
print(x.grad, y.grad)

