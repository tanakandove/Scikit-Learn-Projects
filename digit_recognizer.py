import mnist
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from PIL import Image

#training datasets
x_train = mnist.train_images()
y_train = mnist.train_labels()

#Testing datasets
x_test = mnist.test_images()
y_test = mnist.test_labels()

#print('X_Train', x_train)
#print('X_Test', x_test)


#Resizing the images to 28*28
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))


#Converting the Pixels to between 0 - 1. Neural Networks can learn effectively numbers between 0 and 1
x_train = (x_train/256)
x_test = (x_test/256)

#print(x_train[9])

#Create the model
clf = MLPClassifier(solver="adam", activation="relu", hidden_layer_sizes=(64, 64))

#Train the model
clf.fit(x_train, y_train)

#Make Predictions
predictions = clf.predict(x_test)

#Print to see the size of th data
print(y_test.shape)
print(predictions.shape)


#Perfomance Accurcay: Confusion Matrix
acc = confusion_matrix(y_test,predictions)

#print(acc)

def accuracy(cm):
    diagonal = cm.trace()
    elements = cm.sum()
    return diagonal/elements
print(accuracy(acc))


#Loading the Image
#Convert the pixels within the range 0 - 1
img = Image.open('five1.png')
data = list(img.getdata())

for i in range(len(data)):
    data[i] = 255 - data[i]
five = np.array(data)/256


#Making a prediction of the handwritten image 5
p = clf.predict([five])

#Print out the result
print(p)



