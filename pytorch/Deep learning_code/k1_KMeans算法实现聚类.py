import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# KMeans算法实现聚类
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

data = pd.read_csv('data.csv')

x = data.drop(['labels'], axis=1)
y = data.loc[:, 'labels']

# print(pd.value_counts(y))  # 查看各个类别的个数

# 数据可视化
plt.figure(figsize=(10, 10))
label0 = plt.scatter(x.loc[:, 'V1'][y == 0], x.loc[:, 'V2'][y == 0])
label1 = plt.scatter(x.loc[:, 'V1'][y == 1], x.loc[:, 'V2'][y == 1])
label2 = plt.scatter(x.loc[:, 'V1'][y == 2], x.loc[:, 'V2'][y == 2])
plt.title('title')
plt.xlabel('v1')
plt.ylabel('v2')
plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
# plt.show()

km_model = KMeans(n_clusters=3, random_state=0)
km_model.fit(x)
center = km_model.cluster_centers_
# print("中心点", center)
# print(center[:,0],center[:,1])
plt.scatter(center[:, 0], center[:, 1], color='black')
plt.show()

# 计算准确率
y_predict = km_model.predict(x)
print(accuracy_score(y, y_predict))


# 可视化结果索引
plt.figure(figsize=(10, 10))
label0 = plt.scatter(x.loc[:, 'V1'][y_predict == 0], x.loc[:, 'V2'][y_predict == 0])
label1 = plt.scatter(x.loc[:, 'V1'][y_predict == 1], x.loc[:, 'V2'][y_predict == 1])
label2 = plt.scatter(x.loc[:, 'V1'][y_predict == 2], x.loc[:, 'V2'][y_predict == 2])
plt.title('title')
plt.xlabel('v1')
plt.ylabel('v2')
plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
plt.show()

# 矫正数据
y_corrected = []
for i in y_predict:
    if i == 0:
        y_corrected.append(1)
    elif i == 1:
        y_corrected.append(2)
    else:
        y_corrected.append(0)

print(accuracy_score(y,y_corrected))

# 可视化结果索引

y_corrected = np.array(y_corrected)

plt.figure(figsize=(10, 10))
label0 = plt.scatter(x.loc[:, 'V1'][y_corrected == 0], x.loc[:, 'V2'][y_corrected == 0])
label1 = plt.scatter(x.loc[:, 'V1'][y_corrected == 1], x.loc[:, 'V2'][y_corrected == 1])
label2 = plt.scatter(x.loc[:, 'V1'][y_corrected == 2], x.loc[:, 'V2'][y_corrected == 2])
plt.title('title')
plt.xlabel('v1')
plt.ylabel('v2')
plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
plt.show()