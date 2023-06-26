from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from matplotlib import pyplot as plt

boston = datasets.load_boston()

X = boston.data

y = boston.target

l_reg = linear_model.LinearRegression()

plt.scatter(X.T[5], y)
plt.plot(X[0], y)
plt.show()

