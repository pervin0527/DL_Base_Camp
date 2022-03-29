import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Constant

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

x = tf.constant([[10.]]) # vector가 아니라 matrix 형태로 선언했음.
print(x)

w, b = tf.constant(10.), tf.constant(20.)
print(w, b)

w_init, b_init = Constant(w), Constant(b) # Tensor가 생성되는 것이 아니라 initializer 객체가 생성된다.
print(w_init, b_init)

dense = Dense(units=1,
              activation="linear",
              kernel_initializer=w_init,
              bias_initializer=b_init)

y = dense(x)
W, B = dense.get_weights()

print(f"y : {y.shape}, {y}") # 10 * 10 + 20 = 120
print(f"W : {W.shape}, {W}")
print(f"B : {B.shape}, {B}")