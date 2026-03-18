import tensorflow as tf
import numpy as np
#Rules to follow when building a model


x = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

x = tf.cast(tf.constant(x), dtype=tf.float32)
y = tf.cast(tf.constant(y), dtype=tf.float32)

x =tf.reshape(x, (-1,1))
y =tf.reshape(y, (-1,1))

tf.random.set_seed(42)

#Build a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

#Compile it
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])
#Fit it
model.fit(x, y, epochs=5)

test = tf.constant([[17.0]], dtype=tf.float32)
xrp = model.predict(test)
print(xrp)
