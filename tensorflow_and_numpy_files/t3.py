import tensorflow as tf
import numpy as np

#create a tensor from numpy array
#Difference between numpy and tensorflow is that tensors can be run on a GPU much faster for numerical computing

numpy_A = np.arange(1, 25, dtype=np.int32)
print(numpy_A)

A = tf.constant(numpy_A, shape=(2,3,4))
print(A)

#getting information from tensors
#shape
#rank
#axis or dimension
#size

#create a tensor with 4 dimensions
rank_4 = tf.zeros(shape=([3,4,5,6]))
print(rank_4)
print(rank_4[0])

#getting size of tensor
print(rank_4.shape, rank_4.ndim, tf.size(rank_4))

#Get various attributes of tensor
print("Datatype of every element:", rank_4.dtype)
print("Number of dimensions (rank):", rank_4.ndim)
print("Shape of tensor:", rank_4.shape)
print("Elements along the 0 axis:", rank_4.shape[0])
print("Element along the last axis:", rank_4.shape[-1])
print("Total number of elements in tensor:", tf.size(rank_4).numpy())






