from sklearn import tree

feature = [[140, 1], [130, 1], [150, 0], [170, 0]]
# labels = ["apple","apple","orange","orange"]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature, labels)
print(clf.predict([[150, 0]]))

#
# 二、鸢尾花数据集训练决策树
#
# 下面是根据google机器学习视频，写的鸢尾花数据集的决策树训练、测试、可视化代码
#
# 鸢尾花数据集下载：http: // download.csdn.net / detail / tz_zs / 9874935）
#
# 参考资料：http: // cda.pinggu.org / view / 3074.
# html

# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
# 引入数据集
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
print(iris.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target_names)  # ['setosa' 'versicolor' 'virginica']
print(iris.data[0])  # [ 5.1  3.5  1.4  0.2]
print(iris.target[0])  # 0
test_idx = [0, 50, 100]  # 取0,50，100位置的数据作为测试集（所以这里测试集只有三组数据）

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# print(test_target)
# print(clf.predict(test_data))

# viz code 可视化 制作一个简单易读的PDF
from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
print(len(graph))  # 1
print(graph)  # [<pydot.Dot object at 0x000001F7BD1A9630>]
print(graph[0])  # <pydot.Dot object at 0x000001F7BD1A9630>
# graph.write_pdf("iris.pdf")
graph[0].write_pdf("iris.pdf")
