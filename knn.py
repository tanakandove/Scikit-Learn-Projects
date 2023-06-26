from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


#Reading the data
data = pd.read_csv('car.data')

#print(data.head())

#Data Labels
X = data[[
    'buying',
    'maint',
    'safety'
]].values


#Label to be predicted
y = data[['class']]

#print(X, y)

#converting X
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

#print(X)

#converting y to numbers
label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good':2,
    'vgood':3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)
#print(y)

#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Build the Model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights = 'uniform')

#Train the Model
knn.fit(X_train, y_train)

#Make Predictions
predictions = knn.predict(X_test)

#Calculate Accuracy
accuracy = metrics.accuracy_score(y_test, predictions)

#See the results
print("Predictions:", predictions)
#print("Actual:", y_test)
print("Accuracy:", accuracy)

a = 123
print("actual value:", y[a])
print("predicted value:", knn.predict(X)[a])



