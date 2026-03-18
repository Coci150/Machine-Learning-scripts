import tensorflow as tf
from matplotlib import pyplot as plt

x = tf.range(-100, 100, 4)
y = x + 10

print(len(x))

x_train = x[:40]
y_train = y[:40]

x_test = x[40:]
y_test = y[40:]

tf.random.set_seed(42)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1]),
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              metrics=["mae"])

y_predict = model.predict(x_test)
print(y_predict)
print(y_test)

def plot_pred(train_data=x_train, train_label=y_train,
              test_data=x_test, test_label=y_test,predictions=y_predict):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_label,c="b", label="Training Data")
    plt.scatter(test_data,test_label,c="g", label="Testing Data")
    plt.scatter(test_data,predictions,c="r", label="Predictions")
    plt.legend()
    plt.show()

plot_pred()