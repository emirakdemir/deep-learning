import numpy as np
import matplotlib.pyplot as plt

#vector#
a = np.array(18) #scalar value is 0 dimensional tensor
a
a.ndim #scalar 0 axis

a = np.array([18, 6, 2, 16, 8]) #one-dimensional vector


#matrix#
a = np.array([[33, 87, 21, 10, 7],
             [30, 89, 7, 4, 3],
             [27, 93, 14, 7, 1]]) #matrix is ​​two dimensional tensor
a.ndim


#tensor#
a = np.array([[[33, 87, 21, 10, 7], 
               [30, 89, 7, 4, 3], 
               [27, 93, 14, 7, 1]], 
              [[33, 87, 21, 10, 7], 
               [30, 89, 7, 4, 3], 
               [27, 93, 14, 7, 1]], 
              [[33, 87, 21, 10, 7], 
               [30, 89, 7, 4, 3], 
               [27, 93, 14, 7, 1]]]) 
a.ndim #three-dimensional tensor


#example display from image data
from keras.datasets import mnist
(train_images, train_labels), (test_labels) = mnist.load_data()

train_images.ndim #3
train_images.shape #(60000, 28, 28)
train_images.dtype #uint8

digit = train_images[7]
plt.imshow(digit)
plt.show

mydr = train_images[7:77] #selects samples 7 to 77 and writes them to a directory
mydr.shape # (70, 28, 28)
mydr = train_images[7:77, :, :]
mydr.shape # (70, 28, 28)



#transactions

def naive_relu(x): #relu 
  assert len(x.shape) == 2 
  x = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i, j] = max(x[i, j], 0)
  return x

x = np.array([[33, 87, 21, 10, 7],
             [30, 89, 7, 4, 3],
             [27, 93, 14, 7, 1]])
z = naive_relu(x)
z


def naive_add(x, y):
  assert len(x.shape) == 2
  assert x.shape == y.shape

  x = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i, j] += y[i, j]
  return x

y = np.array([[33, 87, 21, 10, 7],
             [30, 89, 7, 4, 3],
             [27, 93, 14, 7, 1]])
z = naive_add(x, y)
z

x = np.array([[33, 87, 21, 10, 7],
             [30, 89, 7, 4, 3],
             [27, 93, 14, 7, 1]])

y = np.array([[33, 87, 21, 10, 7],
             [30, 89, 7, 4, 3],
             [27, 93, 14, 7, 1]])

z = np.maximum(z, 0.)
z

z = x + y
z

def naive_add_matrix_and_vector(x, y):
  assert len(x.shape) == 2
  assert len(y.shape) == 1
  assert x.shape[1] == y.shape[0]

  x = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i, j] += y[j]
  return x

y = np.array([18, 6, 2, 16, 8]) 
z = naive_add_matrix_and_vector(x, y)
z


x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))

z = np.maximum(x, y)

x = np.array([[33, 87, 21, 10, 7],
             [30, 89, 7, 4, 3],
             [27, 93, 14, 7, 1]])
y = np.array([18, 6, 2, 16, 8])
z = np.dot(x, y)
z


def naive_vector_dot(x, y):
  assert len(x.shape) == 1
  assert len(y.shape) == 1
  assert x.shape[0] == y.shape[0]

  z = 0.

  for i in range(x.shape[0]):
    z += x[i] * y[i]

  return z

x = np.array([18, 6, 2, 16, 8])
y = np.array([18, 6, 2, 16, 8])
z = naive_vector_dot(x, y)
z


def naive_add_matrix_and_vector(x, y):
  assert len(x.shape) == 2 
  assert len(y.shape) == 1
  assert x.shape[1] == y.shape[0]

  x = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i, j] += y[i]

  return x

x = np.array([[33, 87, 21, 10, 7],
             [30, 89, 7, 4, 3],
             [27, 93, 14, 7, 1]])
y = np.array([18, 6, 2, 16, 8])
z = naive_add_matrix_and_vector(x, y)
z


x = np.random.random((64, 3, 32, 10))
y= np.random.random((32, 10))
z = np.maximum(x, y)
z.shape


#inner multiplication
x = np.array([[33, 87, 21, 10, 7],
             [30, 89, 7, 4, 3],
             [27, 93, 14, 7, 1]])
y = np.array([18, 6, 2, 16, 8])
z = np.dot(x, y)
z

def naive_vector_dot(x, y):
  assert len(x.shape) == 1
  assert len(y.shape) == 1
  assert x.shape[0] == y.shape[0]

  z = 0.

  for i in range(x.shape[0]):
    z += x[i] * y[i]

  return z


x = np.array([18, 6, 2, 16, 8])
y = np.array([18, 6, 2, 16, 8])
z = naive_vector_dot(x, y)
z

def naive_matrix_vector_dot(x, y):
  assert len(x.shape) == 2
  assert len(y.shape) == 1
  assert x.shape[1] == y.shape[0]

  z = np.zeros(x.shape[0])

  for i in range(x.shape[0]):
    for i in range(x.shape[1]):
      z[i] += x[i, j] * y[j]

  return z

def naive_matrix_dot(x, y):
  z = np.zeros(x.shape[0])
  for i in range(x.shape[0]):
    z[i] = naive_vector_dot(x[i, :],  y)

  return z

def naive_matrix_dot(x, y):
  assert len(x.shape) == 2
  assert len(y.shape) == 2


  assert x.shape[1] == y.shape[0]

  z = np.zeros((x.shape[0], y.shape[1]))

  for i in range(x.shape[0]):
    for j in range(y.shape[1]):
      row_x = x[i, :]
      column_y = y[:, j]
      z[i, j] = naive_vector_dot(row_x, column_y)
  return z


#Kanal / Şekil Değiştirme
train_images = train_images.reshape(60000, 28 * 28)

x =np.array([[0., 1.],
            [2., 3.],
            [4., 5.]])

x

x = x.reshape((6, 1))
x

x = x.reshape((2, 3))
x

x = np.zeros((100, 10))
x = np.transpose(x)
print(x.shape)