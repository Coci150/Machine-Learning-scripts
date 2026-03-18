import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model

x = tf.range(-100, 100, 4)
y = x + 10

print(len(x))

x_train = x[:40]
y_train = y[:40]

x_test = x[40:]
y_test = y[40:]

#print(len(x_train), len(x_test), len(y_train), len(y_test))
#plt.figure(figsize=(10, 7))
#plt.scatter(x_train, y_train, c="blue", label="Training data")
#plt.scatter(x_test, y_test, c="red", label="Testing data")
#plt.legend()
#plt.show()

tf.random.set_seed(42)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.fit(x_train, y_train, epochs=100, verbose=0)


