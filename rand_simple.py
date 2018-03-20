#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import matplotlib.pyplot as plt
import math
import argparse
import os
import random
import json
import sys

class Model(Chain):
    def __init__(self):
        initW = chainer.initializers.HeNormal()
        super(Model, self).__init__(
            l1 = L.Linear(1, 256, initialW=initW),
            l2 = L.Linear(256, 128, initialW=initW),
            l3 = L.Linear(128, 1, initialW=initW),
        )
        self.output = None
        self.loss = None

    def __call__(self, x, t):
        h = self.predict(x)
        self.loss = F.mean_squared_error(h, t)
        report({'loss': self.loss}, self)
        self.output = h
        return self.loss

    def predict(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h



### CREATE DATASET
# func, start, end, step
def get_raw_dataset(f, s, e, num, gpu=-1):
    x_arr = np.linspace(s, e, num)
    t_arr = f(x_arr)
    x_arr = x_arr.astype(np.float32).reshape(-1,1)
    t_arr = t_arr.astype(np.float32).reshape(-1,1)
    return x_arr, t_arr

def get_random_dataset(f, s, e, num, gpu=-1):
    x_arr, t_arr = get_raw_dataset(f, s, e, num, gpu)
    np.random.shuffle(t_arr)
    return x_arr, t_arr

### OPTIONS
parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--batchsize', '-b', type=int, default=4,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=200000,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--plotfile', '-f', default='plot-',
                    help='File name to output the plot')

parser.add_argument('--rand', '-r', action='store_true',
                    help='Replace all samples with random ones')
args = parser.parse_args()

target_function = lambda x: (np.sin(x) + 1) / 2

N = 16
NT = 1024

train_raw = get_random_dataset(target_function, 0, 2 * math.pi, N)
print("Notice: Train with Random Samples")

x_train, y_train = train_raw


### MAIN
model = Model()

if args.gpu >= 0:
    xp = cuda.cupy
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()
    print('GPU')
else:
    xp = np
# end if

# optimizer = optimizers.NesterovAG(lr=0.1, momentum=0.9)
optimizer = optimizers.Adam()
optimizer.setup(model)

if not os.path.exists(args.out):
    print("Notice: Create New Directory", args.out)
    os.makedirs(args.out)

min_train_loss = 0.1

batchsize = args.batchsize
loss_train = []
for epoch in xrange(1, args.epoch + 1):
    # learning loop
    sum_loss = 0
    num = 0
    model.train = True
    perm = np.random.permutation(N)
    for i in xrange(0, N, batchsize):
        chosen_ids = perm[i:i + batchsize]
        x = chainer.Variable(xp.asarray(x_train[chosen_ids]), volatile='off')
        t = chainer.Variable(xp.asarray(y_train[chosen_ids]), volatile='off')
        optimizer.update(model, x, t)
        loss = float(model.loss.data)
        sum_loss += loss * batchsize
        num += batchsize
    # end for
    train_loss = sum_loss / num
    loss_train.append(train_loss)
    if (epoch % 10 == 0):
        print(json.dumps({'epoch': epoch, 'train loss': train_loss}))

    # minimum check
    if (train_loss < min_train_loss):
        min_train_loss = train_loss
        print('epoch :', epoch, ', minimum train loss :', min_train_loss)



loss_train = np.array(loss_train)
np.save('rand_loss_train.npy', loss_train)


plt.cla()
plt.grid(which='major',color='gray',linestyle='-')
plt.xlabel('epoch')
plt.ylabel('error')
plt.plot(loss_train, label='train loss')
plt.title('Random Train Loss')
plt.legend()
# plt.ylim(0.0001, 0.8)
plt.yscale('log')
# plt.show()
plt.savefig('rand_loss_plot.png', transparent=True)


NT = 1024
x_test = np.linspace(0, 2 * math.pi, NT).astype(np.float32).reshape((-1,1))
out_data = np.zeros((NT,1))
test_bat = 128
model.train = False
for i in xrange(0, NT, test_bat):
    x = chainer.Variable(xp.asarray(x_test[i:i + test_bat]), volatile='off')
    out_data[i:i + test_bat] = cuda.to_cpu(model.predict(x).data)

out_data = out_data.reshape((NT,))
x_test = x_test.reshape((NT,))


print(x_test.shape)
print(out_data.shape)

plt.cla()
plt.grid(which='major',color='gray',linestyle='-')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 2 * math.pi)
plt.ylim(-2, 2)
plt.plot(x_test, out_data, label='test')
plt.plot(train_raw[0], train_raw[1], '.', label='sample value')
plt.title('Random Approximation')
plt.legend()
# plt.show()
plt.savefig('rand_func_plot.png', transparent=True)
