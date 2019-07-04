#-*- coding:utf-8 -*-  
import matplotlib.pyplot as plt  
import matplotlib
import numpy as np

print matplotlib.__version__
def loadData(flieName):  
    inFile = open(flieName, 'r')#以只读方式打开某fileName文件  
  
    #定义两个空list，用来存放文件中的数据  
    train_loss = []  
    accurary = []  
    test_loss = []
    step = []
  
    for line in inFile:  
        trainingSet = line.split() #对于每一行，按' '把数据分开，这里是分成两部分  
        train_loss.append(trainingSet[2]) #第一部分，即文件中的第一列数据逐一添加到list X 中  
        step.append(trainingSet[1]) #第二部分，即文件中的第二列数据逐一添加到list y 中  
        accurary.append(trainingSet[3])
        # test_loss.append(trainingSet[3])
    inFile.close()
    return  train_loss, accurary, test_loss, step


train_loss, accurary, test_loss, step = loadData('log.txt')
plt.xlabel('Step')
# plt.plot(step, train_loss, 'g', label='train loss')
plt.plot(step , accurary, 'r', label='accurary')
# plt.plot(step , test_loss, 'b', label='test_loss')

#plt.plot(step, train_loss, 'g', label='train loss')
'''plt.plot(step , accurary, 'r', label='accurary')
plt.plot(step , test_loss, 'b', label='test_loss')'''
plt.legend()
plt.grid(True)
plt.show()