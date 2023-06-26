import mnist
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from PIL import Image

#training
x_train = mnist.train_images()
y_train = mnist.train_labels()


x_test = mnist.test_images()
y_test = mnist.test_labels()

#print('X_Train', x_train)
#print('X_Test', x_test)


x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))


x_train = (x_train/256)
x_test = (x_test/256)

#print(x_train[9])

clf = MLPClassifier(solver="adam", activation="relu", hidden_layer_sizes=(64, 64))

clf.fit(x_train, y_train)


predictions = clf.predict(x_test)

print(y_test.shape)

print(predictions.shape)


acc = confusion_matrix(y_test,predictions)

#print(acc)

#def accuracy(cm):
    #diagonal = cm.trace()
    #elements = cm.sum()
    #return diagonal/elements
#print(accuracy(acc))


img = Image.open('five1.png')
data = list(img.getdata())

for i in range(len(data)):
    data[i] = 255 - data[i]


five = np.array(data)/256


p = clf.predict([five])

print(p)



