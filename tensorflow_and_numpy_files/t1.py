import tensorflow as tf

#create a tensor
#a scalar is a single number
scalar = tf.constant(7)
print(scalar)

#check dimension of tensor
print(scalar.ndim)

#create a vector
#vector a number with direction (e.g. wind, speed, and direction)
vector = tf.constant([10, 10])
print(vector)

#check dimension of vector
print(vector.ndim)

#creating a matrix (has more than one dimension)
#a matrix is a 2-dimensional array of numbers
matrix = tf.constant([[10, 7], [7, 10]])
print(matrix)

#check dimension of matrix
print(matrix.ndim)

#create another matrix
matrix2 = tf.constant([[10., 7.], [3., 2.], [8., 7.]],
                      dtype=tf.float16)# specify the data type with dtype parameter
print(matrix2)

#create a tensor
#tensor an n-dimensional array of numbers (when n can be any number, a 0-dimensional tensor is a scalar, a 1-dimensional tensor is a vector)
tensor = tf.constant([[[1, 2, 3,],
                      [4, 5, 6,]],
                     [[7, 8, 9],
                      [10, 11, 12,]],
                     [[13, 14, 15,],
                      [16, 17, 18]]])
print(tensor)


