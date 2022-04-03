#
# <h1 align="center">Implementation of Logestic Regression</h1>
# <h6 align="center">Excersice 2</h6>
# </br>
# <h3 align="center">Sina Dalvand</h3>
# <h3 align="center">40011415053</h3>
# <h6 align="center">dalvandsina@yahoo.com</h6>
#
#

import numpy as np
import pandas as pd
import random

df = pd.read_excel('dataset.xls', 'Data').to_numpy()

threshold = int(len(df) * 0.7)
randomArray = [True if i < threshold else False for i in range(len(df))]
random.shuffle(randomArray)
randomArray = list(zip(randomArray, df.tolist()))
test_set = np.array(list(map(lambda y: y[1], filter(lambda x: x[0] == False, randomArray))))
train_set = np.array(list(map(lambda y: y[1], filter(lambda x: x[0] == True, randomArray))))

featureCounts = len(df[0])-1
X_train = train_set[:, :featureCounts]
Y_train = train_set[:, featureCounts]
X_test = test_set[:, :featureCounts]
Y_test = test_set[:, featureCounts]


def normalize(data):
    data = data.astype(float)
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


X_train = normalize(X_train)
X_test = normalize(X_test)
X_train = np.column_stack((np.ones((len(X_train), 1)), X_train))
X_test = np.column_stack((np.ones((len(X_test), 1)), X_test))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def specificClass(y, j):
    return list(map(lambda x: [1] if (x[0] == j) else [0], y))


def gradientDescent(X, y, alpha, itr):
    n = X.shape[1]
    classes = list(set(y))
    thetas = []
    for j in range(len(classes)):
        theta = np.zeros((n, 1))
        innerY = np.asarray(list(map(lambda x: [1] if (x[0] == classes[j]) else [0], y)))
        m = len(innerY)
        for i in range(itr):
            theta = theta - (alpha * (1 / m * (np.dot(X.T, (sigmoid(np.dot(X, theta)) - innerY)))))
        thetas.append(theta.flatten())
    return thetas, classes


thetas, classes = gradientDescent(X_train, Y_train, 0.03, 500)


def predict(X, theta):
    p = sigmoid(X @ np.asarray(theta).T)
    p = np.asarray(list(map(lambda x: np.argmax(x), p)))
    return p


train_set_value = list(map(lambda x: classes[x], predict(X_train, thetas)))
train_set_percent = sum(train_set_value == Y_train) / len(Y_train)
print(f"Accuracy for Train Set: {train_set_percent}")

test_set_value = list(map(lambda x: classes[x], predict(X_test, thetas)))
test_set_percent = sum(test_set_value == Y_test) / len(Y_test)
print(f"Accuracy for Test Set: {test_set_percent}")
