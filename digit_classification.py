#Digit classification with artificial neural networks using mnist dataset


#load the dataset
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape
len(train_images)
train_labels
test_images.shape
test_labels


#neural network architecture
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


#compilation of neural network model
network.compile(optimizer = 'rmsprop',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

#preparation of entries and labels
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#training the neural network model
network.fit(train_images, train_labels, epochs = 5, batch_size = 128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
'test_loss:', test_loss
'test_acc:', test_acc