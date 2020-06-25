### QUICK NOTES ON TENSORFLOW ###

# %tensorflow_version 2.x (add this line if compiling in google colab)
# checks tf installation and version
import tensorflow as tf
print (tf.version)

# different data types with tensors (scalar) (rank 0)
string = tf.Variable("string type", tf.string)
number = tf.Variable(123, tf.int16)
floating = tf.Variable(5.89, tf.float64)

# 1 dimension
rank1_tensor = tf.Variable(["test", "test2", "test3"], tf.string)

# 2 dimensions (matrices)
rank2_tensor = tf.Variable([["test", "test2"], ["test3", "test4"], ["test5", "test6"]], tf.string)

# returns the rank of the tensor
print("tensor 1 rank:")
print(tf.rank(rank1_tensor))
print("\ntensor 2 rank:")
print(tf.rank(rank2_tensor))

# returns shape of tensor (if rank 2: 1st num = list amount, 2nd num = elements inside)
print("\ntensor 1 shape:")
print(rank1_tensor.shape)
print("\ntensor 2 shape:")
print(rank2_tensor.shape)
print("\n")

# changing tensor shape
tensor1 = tf.ones([1,2,3]) # creates a tensor of shape [1,2,3] of ones
tensor2 = tf.reshape(tensor1, [2,3,1]) # reshapes to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1]) # -1 will calculate the necessary dimension in that place
                                       # it would reshape it to [3,2]
print(tensor1)
print(tensor2)
print(tensor3)

