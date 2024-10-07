from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建分类器
models = []
models.append(("LR", LogisticRegression()))  # 逻辑回归
models.append(("NB", GaussianNB()))  # 高斯朴素贝叶斯
models.append(("RF", RandomForestClassifier(n_estimators=10, random_state=42)))  # 随机森林分类器
models.append(("DT", DecisionTreeClassifier()))  # 决策树分类
models.append(("SVM", SVC()))  # 支持向量机分类
models.append(("adboost", AdaBoostClassifier()))  # adboost

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=None, shuffle=False)
    cv_result = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    names.append(name)
    results.append(cv_result)

for i in range(len(names)):
    print(names[i], results[i].mean())
