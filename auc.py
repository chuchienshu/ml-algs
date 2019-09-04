
'''
author: chuchienshu
date: 2019/9/4

description: compute roc-auc for binary classification. ref to https://www.jianshu.com/p/c61ae11cc5f6
TPR / recall / sensitivity = TP/(TP + FN)
FPR  = FP / (FP + TN)
'''
import matplotlib.pyplot as plt

def roc_auc(label_gt_pairs):
  l_gt = sorted(label_gt_pairs, key=lambda x:x[1], reverse=True)

  # TP, FP, TN, FN = 0, 0, 0, 0
  FPR_TPR = []

  for _, threashold in l_gt:
    TP, FP, TN, FN = 0, 0, 0, 0
    for label, score in l_gt:
      if score >= threashold:
        if label == 1:
          TP += 1
        else:
          FP += 1
      else:
        if label == 0:
          TN += 1
        else:
          FN += 1
    
    tpr = 1.0*TP/(TP + FN)
    fpr = 1.0*FP / (FP + TN)
    FPR_TPR.append([tpr, fpr])

  x = [p[1] for p in FPR_TPR]
  y = [p[0] for p in FPR_TPR]

  auc = 0.
  for i in range(1, len(FPR_TPR)):
    TPR2, FPR2 = FPR_TPR[i]
    _, FPR1 = FPR_TPR[i-1]
    auc += (FPR2-FPR1)*TPR2
  
  plt.figure()
  plt.plot(x,y)
  plt.show()

  return auc
    



a = [[1, 0.9],
    [0, 0.7],
    [1, 0.6],
    [1, 0.55],
    [0, 0.52],
    [1, 0.4],
    [0, 0.38],
    [0, 0.35],
    [1, 0.31],
    [0, 0.1]]

roc_auc(a)