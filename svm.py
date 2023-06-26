#source : freeCodeCamp

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


#Loading the data
iris = datasets.load_iris()

#Classes of the Iris plant
classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

#Training and testing dataset
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Build the model
model = svm.SVC()

#Train the Model
model.fit(X_train, y_train)

#Make Predictions
predictions = model.predict(X_test)

#Perfomance Metrics:
accuracy = metrics.accuracy_score(y_test, predictions)

#Print the results
print("Predictions :", predictions)
print("Actual      :", y_test)
print("Accuracy :", accuracy)
print(model)

