'''
Copyright reserved.
author: chuchienshu 2018/6/1
'''
print("fsadf")



import numpy as np
from math import exp
from random import seed
from random import random

from dataset import load_data

train_set = load_data('train', shuffle=True)
test_set = load_data('test', shuffle = True)
train_len = len(train_set)
test_len = len(test_set)
'''
Ts = open('ts.txt', 'a')
for t in train_set:
    Ts.write('%f' % t[1])

print(len(train_set))
exit()
'''

class sigmoid(object):
    def __init__(self):
        pass

    def forward(self, inputs):
        self.out = 1 / (1 + np.exp(-1 * inputs))
        return self.out

    def bp(self, grad_in):
        grad_out = self.out * (1 - self.out) * grad_in
        return grad_out


class full_connec(object):
    def __init__(self, n_inputs, n_outs):
        self.weight = np.random.normal(size=(n_inputs, n_outs))

    def forward(self, inputs):
        self.inputs = inputs
        # self.input = input
        out = np.matmul(self.inputs, self.weight)
        return out

    def bp(self, grad_input, lr):
        grad_out = np.matmul(grad_input, self.weight.transpose(1, 0))
        self.update_weight(grad_input, lr=lr)
        return grad_out

    def update_weight(self, grad_input, lr):
        grad_currlayer = np.matmul(self.inputs.transpose(1, 0), grad_input)
        self.weight -= lr * grad_currlayer


class NetWork(object):
    def __init__(self):
        self.fc_1 = full_connec(200, 100)
        self.sg = sigmoid()
        # self.fc = full_connec(100,100)
        self.fc_2 = full_connec(100, 1)
        self.sg1 = sigmoid()

    def forword(self, inputs):
        x = self.fc_1.forward(inputs)
        x = self.sg.forward(x)
        # x = self.fc.forward(x)
        x = self.fc_2.forward(x)
        x = self.sg1.forward(x)
        return x

    def bp(self, grad_in, lr):
        y = self.sg1.bp(grad_in)
        y = self.fc_2.bp(grad_in, lr=lr)
        # y = self.fc.bp(y, lr=lr)
        y = self.sg.bp(y)
        self.fc_1.bp(y, lr=lr)


# iput = np.ones((1,6))


class MSE_loss(object):
    def __init__(self):
        pass

    def forward(self, output, label):
        if not label == 1:
            self.grad = (output - label)  *0.75
        else:
            self.grad = (output - label)  *0.25
        return self.grad

    def bp(self):
        return self.grad


loss = MSE_loss()
net = NetWork()

epoch = 50
lr = 0.01
for e in range(epoch):
    if (e + 1 )%5 == 0:
        lr /= 10
    trainlog = open('trainlog.txt','a')
    totalloss = 0.0
    print('training')
    for i in range(train_len):
        inp = np.expand_dims(train_set[i][0], axis=0)
        out = net.forword(inp)
        loss_value = loss.forward(out, np.array(train_set[i][1]))
        totalloss += loss_value
        print(totalloss)
        # print(loss_value)
        net.bp(loss_value, lr = lr)
    
    trainlog.write('epoch %d %f \n' % (e, totalloss))
    if (e + 1) % 1 == 0:
        log = open('log.txt', 'a')
        print('testing')
        count = 0
        for i in range(test_len):
            tinp = np.expand_dims(test_set[i][0], axis=0)
            out = net.forword(tinp)
            if out > 0.5 and test_set[i][1] == 1.0:
                count += 1
            if out <= 0.5 and test_set[i][1] == 0.0:
                count += 1
        log.write('epoch %d accrucy %f error %f \n' % (e, count/test_len, 1- count/test_len))

            


