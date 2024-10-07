import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集的一个子集（仅包含两个类别）
iris = datasets.load_iris()
X = iris.data[iris.target != 2]  # 移除弗吉尼亚鸢尾花
y = iris.target[iris.target != 2]
y = np.where(y == 0, -1, 1)  # 将标签转换为-1和1（SVM通常使用这样的标签）

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放（可选，但通常对SVM有帮助）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建SVM分类器
svm = SVC(kernel='linear', C=1.0, random_state=42)

# 训练SVM模型
svm.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = svm.predict(X_test)
print('y_pred:',y_pred)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)