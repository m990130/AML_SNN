import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

N = 64

x0 = torch.randn(N, 1)
x = Variable(x0)
y = Variable(x0, requires_grad=False)

A = Variable(torch.randn(1, 1), requires_grad=True)
b = Variable(torch.randn(1), requires_grad=True)

optimizer = optim.SGD([A, b], lr=1e-1)
for t in range(10):
    print('---------------------------------------')
    optimizer.zero_grad()
    # print A.grad, b.grad
    y_pred = torch.matmul(x, A) + b
    loss = ((y_pred - y) ** 2).mean()
    print(t, loss.item())
    loss.backward()
    optimizer.step()
# print [A, b]
