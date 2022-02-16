import os
import gzip
import pickle
import argparse
import numpy as np
import urllib.request

def download_file():
    if not os.path.isdir(f"{save_path}/MNIST"):
        os.makedirs(f"{save_path}/MNIST")

    for file_name in file_names.values():
        if os.path.isfile(f"{save_path}/MNIST/{file_name}"):
            print(f"{file_name} is Already Exist. SKIPPED")
            pass

        else:
            print(f"Downloading {base_url}/{file_name}...", end=' ')
            urllib.request.urlretrieve(f"{base_url}/{file_name}", f"{save_path}/MNIST/{file_name}")
            print("Done")

    print()

def load_img(file_name):
    file_path = f"{save_path}/MNIST/{file_name}"
    
    print(f"Converting {file_name} to Numpy Array")
    with gzip.open(file_path, 'rb') as  f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    data = data.reshape(-1, 784)
    
    return data

def load_label(file_name):
    file_path = f"{save_path}/MNIST/{file_name}"

    print(f"Converting {file_name} to Numpy Array")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return labels

def convert_file():
    dataset = {}
    dataset['train_img'] = load_img(file_names['train_img'])
    dataset['train_label'] = load_label(file_names['train_label'])
    dataset['test_img'] = load_img(file_names['test_img'])
    dataset['test_label'] = load_label(file_names['test_label'])

    return dataset

def init_mnist():
    download_file()
    dataset = convert_file()

    print("Creating Pickle File...", end=' ')
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)

    print("Done")

def change_one_hot_label(x):
    encoded = np.zeros((x.size, 10))
    for idx, row in enumerate(encoded):
        row[x[idx]] = 1

    return encoded

def load_mnist(normalize=True, flatten=True, one_hot_label=True):
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label'], dataset['test_img'], dataset['test_label'])


parser = argparse.ArgumentParser(description="Download mnist dataset")
parser.add_argument("--save-path", type=str, help="path to save mnist dataset")
args = parser.parse_args()

save_path = args.save_path
save_file = f"{save_path}/MNIST/mnist.pkl"
base_url = "http://yann.lecun.com/exdb/mnist"
file_names = {'train_img' : 'train-images-idx3-ubyte.gz',
              'train_label' : 'train-labels-idx1-ubyte.gz',
              'test_img' : 't10k-images-idx3-ubyte.gz',
              'test_label' : 't10k-labels-idx1-ubyte.gz'}

if __name__ == "__main__":
    init_mnist()
    train_images, train_labels, test_images, test_labels = load_mnist(flatten=False)
    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)