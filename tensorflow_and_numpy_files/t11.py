import tensorflow as tf
from matplotlib import pyplot as plt

#tensor1 = tf.constant(["room2","room"])
#print(tensor1)

x_train = tf.constant([[1,4],[2,7],[3,10],[4,13],[5,16]])
y_train = tf.constant([[-1,-4],[-2,-7],[-3,-10],[-4,-13],[-5,-16]])
print(x_train)
print(y_train)


tf.random.set_seed(42)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
model.compile(loss=tf.keras.losses.mae,)

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),metrics=['mae'])

model.fit(x_train,y_train, epochs=100)
test = tf.constant([6,19])
y_pred = model.predict(test)
print(y_pred)


