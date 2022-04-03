'''Featurer가 1개일 때 Affine function을 구현해본다.'''
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sample = tf.constant([1, 2, 3, 4, 5], shape=5)
print(sample.shape, sample)

sample_x = tf.constant([[10., 20.]]) # 개념 설명 때 feature가 1개인 데이터들로 구성된 coulmn vector를 떠올리면 된다.
print(sample_x.shape, sample_x) # (1, 2), [[10., 20.]]

x = tf.constant([[10.]]) # vector가 아니라 matrix 형태로 선언했음.
print(x.shape, x) # (1, 1), [[10.]]

"""
아래 Dense layer에 대해 weight와 bias 값을 별도로 초기화 선언하지 않았는데,
이는 입력 되는 때에 tensor의 shape을 보고 자동으로 초기화 해준다.
"""
dense = Dense(units=1, activation="linear") # affine function 정의.

y = dense(x) # x를 affine function에 입력한 값을 y에 저장.
W, B = dense.get_weights()
print(W, B, '\n')

# dense = Dense(units=1, activation='linear')
# tmp = dense.get_weights()
# print(tmp)

""" 연산 검증. """
manual_y = tf.linalg.matmul(x, W) + B

print("===== Input / Weight / Bias =====")
print(f"x : {x.shape}, {x}")
print(f"w : {W.shape}, {W}")
print(f"b : {B.shape}, {B} \n")


print(" ===== Outputs =====")
print(f"y from tf : {y.shape}, {y}")
print(f"y from manual : {manual_y.shape}, {manual_y}")