# Function: K Means
# author chuchienshu
# date 19/4/3
import numpy as np
import matplotlib.pyplot as plt
import random

MAX_ITERATIONS = 100


def getRandomCentroids(x, y , k):
    assert k > 0
    seeds = {}
    for indx in range(k):
        # seed_x = random.sample(list(x), 1)
        # seed_y = random.sample(list(y), 1)
        seed_x = random.randint(0, 10)
        seed_y = random.randint(0, 10)
        # seed_xy = np.concatenate([seed_x, seed_y], axis=0)
        
        seeds[indx] = np.array([seed_x, seed_y])
    return seeds

def cmp(a, b):
    # k_diff = a.keys() - b.keys()
    # v_diff = a.values() - b.values()
    for a_v, b_v in zip(a.values(), b.values()):
        if not np.array_equal(a_v, b_v):
            return False
    
    return  True

# Function: Should Stop
def shouldStop(oldCentroids, centroids, iterations):
    if iterations > MAX_ITERATIONS: 
        return True
    if oldCentroids == None:
        return False
    return cmp(oldCentroids, centroids)

# Function: Get Labels, and sorted_dataset
def getLabels(dataSet, centroids, k):
    bins = []
    sum_axis = dataSet.ndim
    for key, cent in centroids.items():
        # cal = np.sqrt((( dataSet-cent)**2).sum(axis=1))
        distance = np.sqrt((( dataSet-cent)**2).sum(axis=tuple(range(1, sum_axis)))) # keep the extending ability
        bins.append(distance)
    bins = np.stack (bins, axis=1)
    labs = np.argmin(bins ,axis=1)

    sorted_dataset = {}
    for i in range(k):
        _lab = np.where( labs == i)
        sorted_dataset[i] = dataSet[_lab]

    return labs, sorted_dataset



# Function: Get New Centroids
def getCentroids(sorted_dataset, oldCentroids):
    centroids = {}
    for key, cen in sorted_dataset.items():
        if len (cen) == 0:# in case some class missing
            centroids[key] = oldCentroids[key]
            continue
        centroids[key] = np.mean(cen, axis= 0)

    return centroids

def plot_date(centroids, s_data, k, iterration):
    for i in range(k):
        plt.scatter(centroids[i][0], centroids[i][1],s= 100, marker='x')

    markers = ['^', 'o', '8', 'p', '>', 'h']
    for k, val in s_data.items():
        x, y = np.split(val, 2, axis=1)
        plt.scatter(x, y, marker=markers[k])
    plt.savefig('./%d.png' % iterration)
    plt.show()

def kmeans(dataSet, k):
	
    # Initialize centroids randomly
    # numFeatures = dataSet.getNumFeatures()
    centroids = getRandomCentroids(x,y,k)
    iterations = 0
    oldCentroids = None
    
    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations):
        
        oldCentroids = centroids
        iterations += 1
        
        labs, s_data = getLabels(dataSet, oldCentroids, k)
        plot_date(oldCentroids, s_data, k, iterations)
        centroids = getCentroids(s_data , oldCentroids)
    print('iterations %d ' % iterations)
    return centroids, s_data


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([8, 3, 4, 8, 0, 8, 5, 6, 8])
k = 5 # k should be below than len(markers)

centroids, s_data = kmeans(np.stack([x,y], axis=1), k)
print(centroids)
print(s_data)

# plot the final centroids that marker as "X"



