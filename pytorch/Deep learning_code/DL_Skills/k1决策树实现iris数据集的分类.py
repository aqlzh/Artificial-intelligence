import pandas as pd
import numpy as np

# 决策树模型
from sklearn import tree

from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

# 加载数据
data = pd.read_csv('iris_data.csv')

x = data.drop(['target', 'label'], axis=1)
y = data.loc[:, 'label']
# print(x.shape, y.shape)

# 训练模型
dc_tree = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1)
dc_tree.fit(x, y)

y_pred = dc_tree.predict(x)
print(accuracy_score(y, y_pred))

# 画图
plt.figure(figsize=(30, 30))
tree.plot_tree(dc_tree, filled=True,
               feature_names=['sepal length', 'sepal width', 'petal length', 'petal width']
               , class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
# plt.show()
# 下载图片
plt.savefig('iris_tree2.png')