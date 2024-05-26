import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if os.path.getsize('result.pkl') > 0:
    with open('result.pkl', 'rb') as f:
        data = pickle.load(f)

resultlist = []
for result in data:
    result = result.tolist()
    resultmax = max(result)
    resultid = result.index(resultmax)
    resultlist.append(resultid)

labellist = np.load('labels.npy').tolist()
hitcount = [0]*120
totalcount = [0]*120
for i, label in enumerate(labellist):
    totalcount[label] += 1
    if resultlist[i] == label:
        hitcount[label] += 1

accuracy = [hit/total if total!= 0 else 0 for hit, total in zip(hitcount, totalcount)]
plt.bar(range(120), accuracy)
plt.show()
