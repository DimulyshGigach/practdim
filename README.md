# practdim
import os
os.chdir('/content/drive/MyDrive/Colab Notebooks')
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
from sklearn.model_selection import train_test_split
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 10000,train_size = 60000,random_state = 123)
print('Shape of X train:', X_train.shape)
print('Shape of y train:', y_train.shape)
plt.imshow(X_train[123], cmap=plt.get_cmap('gray'))
plt.show()
print(y_train[123])
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels) / 255
X_test = X_test.reshape(X_test.shape[0], num_pixels) / 255
print('Shape of transformed X train:', X_train.shape)
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print('Shape of transformed y train:', y_train.shape)
num_classes = y_train.shape[1]
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=300, input_dim=num_pixels, activation='sigmoid'))
model.add(Dense(units=100, activation='sigmoid'))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print(model.summary())
H = model.fit(X_train, y_train, validation_split=0.1, epochs=100)
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'])
plt.title('Loss by epochs')
plt.show()
scores = model.evaluate(X_test, y_test)
print('Loss on test data:', scores[0])
print('Accuracy on test data:', scores[1])
n = 123
result = model.predict(X_test[n:n+1])
print('NN output:', result)
plt.imshow(X_test[n].reshape(28,28), cmap=plt.get_cmap('gray'))
plt.show()
print('Real mark: ', str(np.argmax(y_test[n])))
print('NN answer: ', str(np.argmax(result)))
from PIL import Image
file_data = Image.open('test.png')
file_data = file_data.convert('L') 
test_img = np.array(file_data)
plt.imshow(test_img, cmap=plt.get_cmap('gray'))
plt.show()
test_img = test_img / 255
test_img = test_img.reshape(1, num_pixels)
result = model.predict(test_img)
print('I think it\'s ', np.argmax(result))
