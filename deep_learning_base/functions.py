import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int64)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    
    return y

def identity_func(x):
    return x

if __name__ == "__main__":
    test2 = np.array([0.3, 2.9, 4.0])
    print(softmax(test2))