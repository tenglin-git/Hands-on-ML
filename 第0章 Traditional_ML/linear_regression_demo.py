import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 1准备数据 y = 2x + 1
train_X = np.linspace(-1, 1, 100)
# print(train_X.shape)
train_Y = 2 * train_X + 1 + np.random.randn(*train_X.shape) * 0.3
# 2可视化数据
plt.plot(train_X, train_Y, 'ro', label="original data")
plt.legend()
plt.show()
# 3训练模型并可视化
model = LinearRegression()
model.fit(train_X.reshape(100, 1), train_Y.reshape(100, 1))
# model.fit(train_X, train_Y)
print("模型的权重和偏置：")
print("权重：", model.coef_)
print("偏置：", model.intercept_)
x = np.array([6])
print("输入6预测结果:", model.predict(x.reshape(1, -1)))
print("输入6计算结果:", model.coef_ * 6 + model.intercept_)

plt.plot(train_X, train_Y, 'ro', label="original data")
plt.plot(train_X, model.predict(train_X.reshape(100, 1)), label="fitted line")
plt.legend()
plt.show()

# 4评估模型
test_X = np.linspace(11, 20, 20)
test_Y = 2 * test_X + np.random.randn(*test_X.shape) * 0.3
print("模型评估score:", model.score(test_X.reshape(20, 1), test_Y.reshape(20, 1)))

# 5模型保存与调用
# from sklearn.externals import joblib
import joblib

print("模型保存与调用")
joblib.dump(model, 'train_model.m')
model = joblib.load('train_model.m')
print("预测结果:", model.predict(x.reshape(1, -1)))
