import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np



mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalizing data. Why? It makes it easier for the network to learn
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test  = tf.keras.utils.normalize(x_test, axis = 1)

y = x_test

plt.imshow(x_test[0], cmap = plt.cm.binary)
plt.show()



model = tf.keras.models.Sequential()

#****Input layer
model.add(tf.keras.layers.Flatten())


model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))


model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer= 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# Training your data:
model.fit(x_train, y_train, epochs = 1)


val_loss, val_acc = model.evaluate(x_test, y_test)
print(x_test, y_test)

model.save('mnist_trained_model')

new_model = tf.keras.models.load_model('mnist_trained_model')

# PREDICTS always takes a list!
predictions = new_model.predict([x_test])