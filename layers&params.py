#layers
from keras import layers
layer = layers.Dense(32, input_shape = (784, ))

#models
from keras import models
model = models.Sequential()
model.add(layers.Dense(32, input_shape=(784, )))
model.add(layers.Dense(32))

model.summary()
"""
Total params: 26,176
Trainable params: 26,176
Non-trainable params: 0
"""