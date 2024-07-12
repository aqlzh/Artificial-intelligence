import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 主成份分析pca
from sklearn.decomposition import PCA

# 数据预处理工具包
from sklearn.preprocessing import StandardScaler  # 标准化处理

data = pd.read_csv('iris_data.csv')
X = data.drop(['target', 'label'], axis=1)
Y = data.loc[:, 'label']
# print(X,Y)

# 建立KNN模型
KNN_MODE = KNeighborsClassifier(n_neighbors=3)
KNN_MODE.fit(X, Y)
y_pred = KNN_MODE.predict(X)
print(accuracy_score(Y, y_pred))

# 标准化处理
x_norm = StandardScaler().fit_transform(X)
# 计算处理后的数据的均值和标准差
x_mean = x_norm.mean()
x_std = x_norm.std()
# 打印均值和标准差
print(X.loc[:, 'sepal length'].mean(), x_mean)
print(X.loc[:, 'sepal length'].std(), x_std)

# 可视化处理
plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.hist(X.loc[:, 'sepal length'], bins=100)
plt.title('sepal length')
plt.subplot(122)
plt.hist(x_norm[:, 0], bins=100)
plt.title('sepal length')
plt.show()

# PCA
pca = PCA(n_components=4)
x_pca = pca.fit_transform(x_norm)
# 打印方差
print(pca.explained_variance_ratio_)
# 可视化操作
plt.figure(figsize=(20, 10))
plt.bar([1, 2, 3, 4], pca.explained_variance_ratio_)
plt.xticks([1, 2, 3, 4], ['pca1', 'pca2', 'pca3', 'pca4'])
plt.show()
# plt.savefig('xxx.png')

# 只保留2个维度
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_norm)
# print(x_pca.shape)

# 可视化
plt.figure(figsize=(20, 10))
setosa = plt.scatter(x_pca[:, 0][Y == 0], x_pca[:, 1][Y == 0])
ver = plt.scatter(x_pca[:, 0][Y == 1], x_pca[:, 1][Y == 1])
vir = plt.scatter(x_pca[:, 0][Y == 2], x_pca[:, 1][Y == 2])
plt.legend((setosa, ver, vir), ('setosa', 'ver', 'vir'))
plt.show()

# 建立KNN模型
KNN_MODE = KNeighborsClassifier(n_neighbors=3)
KNN_MODE.fit(x_pca, Y)
y_pred = KNN_MODE.predict(x_pca)
print(accuracy_score(Y, y_pred))
