from sklearn.decomposition import PCA   
import numpy as np
import os

def dataset(path, list_name):  
    fil_lis = sorted(os.listdir(path))
    for file in fil_lis:  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            dataset(file_path, list_name)  
        else:  
            data = np.loadtxt(file_path)
            list_name.append(data)  
            print(file_path)
    # list_name = sorted(list_name)
    return list_name

trainset_root = '/home/chuchienshu/Documents/研一上/pr/PR_dataset/train/pos'

ts = dataset(trainset_root, [])
ts = np.array(ts)
print(ts.shape)

# data = np.loadtxt(trainset)
# data = np.expand_dims(data, axis = 0)

pca=PCA(n_components=200)  

newData=pca.fit_transform(ts)  
print(newData.shape)  
np.savetxt('train_pos.txt', newData)

print(len(ts))
