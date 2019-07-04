import numpy as np
import random
# trainset = np.loadtxt(root)

trainset = ['/home/chuchienshu/Ccode/pr/train_pos.txt',
            '/home/chuchienshu/Ccode/pr/train_neg.txt']

testset = ['/home/chuchienshu/Ccode/pr/test_pos.txt',
           '/home/chuchienshu/Ccode/pr/test_neg.txt']

def load_data(dtype, shuffle=False):
    if dtype == 'train':
        datas = trainset
    else:
        datas = testset
    input_and_label = []
    for dataroot in datas:
        if 'pos' in dataroot:
            label = 1.0
        else:
            label = 0.0

        dataset = np.loadtxt(dataroot)

        for d in dataset:
            tmp = []
            tmp.append(d)
            tmp.append(label)
            input_and_label.append(tmp)
    if shuffle:
        random.shuffle(input_and_label)

    return input_and_label
    
# def next_data(dataset):


