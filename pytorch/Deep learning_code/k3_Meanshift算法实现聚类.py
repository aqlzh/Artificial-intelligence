import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Meanshift算法实现聚类(estimate_bandwidth 估计半径的方法)
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import accuracy_score

data = pd.read_csv('data.csv')

x = data.drop(['labels'], axis=1)
y = data.loc[:, 'labels']

r = estimate_bandwidth(x, n_samples=50000)
# print(r)

ms_model = MeanShift(bandwidth=r)
ms_model.fit(x)
y_pred = ms_model.predict(x)
print(pd.value_counts(y_pred))
print(pd.value_counts(y))

# 数据矫正（0->2，1->1,2->0）
y_corrected = []
for i in y_pred:
    if i == 0:
        y_corrected.append(2)
    elif i == 1:
        y_corrected.append(1)
    else:
        y_corrected.append(0)

print(accuracy_score(y, y_corrected))
