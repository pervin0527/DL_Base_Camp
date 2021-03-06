import os
import tensorflow as tf
from tensorflow.keras.layers import Dense

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

sample = tf.random.uniform(shape=[10], minval=0, maxval=1) # vector
print(sample.shape, sample)

x = tf.random.uniform(shape=(1, 10), minval=0, maxval=10)
print(x.shape, '\n', x) # (1, 10) row vector 형태. 10개의 feature에 해당하는 값들이 row vector 형태로 matrix로 선언됨.

dense = Dense(units=1)

y = dense(x)
W, B = dense.get_weights() # (10, 1), (1,)

y_manual = tf.linalg.matmul(x, W) + B

print("===== Input / Weight / Bias =====")
print(f"x : {x.shape} \n {x}")
print(f"w : {W.shape} \n {W}")
print(f"b : {B.shape} \n {B} \n")


print(" ===== Outputs =====")
print(f"y from tf : {y.shape}, {y}")
print(f"y from manual : {y_manual.shape}, {y_manual}")