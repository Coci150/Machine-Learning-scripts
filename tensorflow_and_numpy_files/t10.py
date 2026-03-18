import tensorflow as tf
import numpy as np

x = tf.constant([11, 14, 17, 20, 23, 26])
y = tf.constant([-20, -23, -26, -29, -32, -35])
print(x)
print(y)


tf.random.set_seed(42)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=(1,)),
                                    tf.keras.layers.Dense(1, input_shape=(1,)),
                                    tf.keras.layers.Dense(1, input_shape=(1,))])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.101)
              ,metrics=['mae'])

model.fit(x,y,epochs=100)
test = tf.constant([29])
xrp = model.predict(test)
print(xrp)