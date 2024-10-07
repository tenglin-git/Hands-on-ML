import numpy as np


# Sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Sigmoid 激活函数的导数
def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)  # 假设隐藏层有4个神经元
        self.weights2 = np.random.rand(4, 1)  # 假设输出层有1个神经元
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # 应用链式法则计算对权重的偏导数
        # 误差在输出层
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))

        # 误差在隐藏层
        error_layer1 = (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                               self.weights2.T) * sigmoid_derivative(self.layer1))
        d_weights1 = np.dot(self.input.T, error_layer1)

        # 更新权重
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, epochs):
        for i in range(epochs):
            self.feedforward()
            self.backprop()

    def predict(self, X):
        self.layer1 = sigmoid(np.dot(X, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return np.round(self.output)

# 示例数据和标签

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

# 创建神经网络实例并训练
nn = NeuralNetwork(X, y)
nn.train(15000)

# 预测新数据
predictions = nn.predict(np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]))
print(predictions)