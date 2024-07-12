import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

# 计算高斯分布的概率密度
from scipy.stats import norm
# 异常检测模型
from sklearn.covariance import EllipticEnvelope

data = pd.read_csv('anomaly_data.csv')

# 可视化
plt.figure(figsize=(12, 12))
plt.scatter(data.loc[:, 'x1'], data.loc[:, 'x2'])
plt.show()

x1 = data.loc[:, 'x1']
x2 = data.loc[:, 'x2']

plt.hist(x1, bins=100)  # 分布直方图
# plt.show()

plt.hist(x2, bins=100)
# plt.show()

# 求x1与x2的均值和标准差
x1_mean = x1.mean()
x1_std = x1.std()
x2_mean = x2.mean()
x2_std = x2.std()

# 计算高斯分布的概率密度
x1_range = np.linspace(0, 20, 300)
x1_normal = norm.pdf(x1_range, x1_mean, x1_std)
# print(x1_normal)
# 可视化
plt.figure(figsize=(12, 12))
plt.plot(x1_range, x1_normal)
# plt.show()

# 计算高斯分布的概率密度
x2_range = np.linspace(0, 20, 300)
x2_normal = norm.pdf(x2_range, x2_mean, x2_std)
# print(x1_normal)
# 可视化
plt.figure(figsize=(12, 12))
plt.plot(x1_range, x1_normal)
# plt.show()

# 训练模型
ad_model = EllipticEnvelope(contamination=0.02)  # contamination修改概率阈值
ad_model.fit(data)

# 进行预测
y_pred = ad_model.predict(data)
print(pd.value_counts(y_pred))

# 画异常点
plt.figure(figsize=(12, 12))
plt.scatter(data.loc[:, 'x1'], data.loc[:, 'x2'], marker='x')
plt.scatter(data.loc[:, 'x1'][y_pred == -1], data.loc[:, 'x2'][y_pred == -1], marker='o', facecolor='none',
            edgecolor='red', s=150)
plt.show()
