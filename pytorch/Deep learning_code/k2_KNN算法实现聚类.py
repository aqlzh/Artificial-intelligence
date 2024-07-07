import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# KNN算法实现聚类
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('data.csv')

x = data.drop(['labels'], axis=1)
y = data.loc[:, 'labels']

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x, y)
y_pred = knn_model.predict(x)
print(accuracy_score(y, y_pred))
print(knn_model.predict([[80, 60]]))
