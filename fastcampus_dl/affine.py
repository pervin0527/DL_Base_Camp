import tensorflow as tf
from tensorflow.keras.layers import Dense

# x = tf.constant([[10.], [20.]]) # vector가 아닌 matrix 형태로 만들었음. mini-batch를 위해 Column vector 형태로 만든다.
# print(x.shape)

x = tf.constant([[10.]]) # input value
print(x)

dense = Dense(units=1, activation='linear') # Affine function

y_tf = dense(x) # z = xw + b, forward propagation + params initialization
print(y_tf)

W, B = dense.get_weights()
print(W, B)