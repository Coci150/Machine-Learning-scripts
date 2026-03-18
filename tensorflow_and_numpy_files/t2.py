import tensorflow as tf

tensor_c = tf.Variable([10,7])
tensor_unc = tf.constant([10, 7])
print(tensor_c, tensor_unc)

tensor_c[0].assign(7)
print(tensor_c)

#random tensors are tensors of some arbitrary size which contain random numbers
random1 = tf.random.Generator.from_seed(42)# set seed for reproducibility
random1 = random1.normal(shape=(3,2))

random2 = tf.random.Generator.from_seed(42)
random2 = random2.normal(shape=(3,2))
print(random1)
print(random2)
print(random1 == random2)

#shuffle tensor (valuable for when you want to shuffle your data so that the inherent order does not affect learning)
nt_shuffled = tf.constant([[10, 7],
                           [3, 5],
                           [6, 2]])
print(nt_shuffled)
print(nt_shuffled.ndim)

#global level random seed
tf.random.shuffle(nt_shuffled)
print(tf.random.shuffle(nt_shuffled))

#operational level random seed
tf.random.shuffle(nt_shuffled, seed=42)
print(tf.random.shuffle(nt_shuffled, seed=42))

#create a tensor of all ones
tf.ones([10, 7])
print(tf.ones([10, 7]))

#create a tensor of all zeros
tf.zeros([4, 4])
print(tf.zeros([4,4]))




