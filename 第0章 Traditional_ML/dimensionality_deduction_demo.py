import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
data = iris.data
target = iris.target
# 使用PCA进行降维
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)
# 降维后的数据和目标变量；成分Component
df = pd.DataFrame(data_2d, columns=['Component1', 'Component2'])
df['Target'] = target

# 可视化降维后的数据
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for target, color in zip([0, 1, 2], colors):
    subset = df[df['Target'] == target]
    plt.scatter(subset['Component1'], subset['Component2'], label=iris.target_names[target], c=color)

plt.xlabel('Principal Component1')
plt.ylabel('Principal Component2')
plt.legend(title='Target')
plt.title('PCA of Iris Data')
plt.show()