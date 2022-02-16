import os
import urllib.request

def download_file():
    if not os.path.isdir(f"{save_path}/MNIST"):
        os.makedirs(f"{save_path}/MNIST")

    for file_name in file_names:
        print(f"Downloading {base_url}.{file_name}")
        urllib.request.urlretrieve(f"{base_url}/{file_name}", f"{save_path}/MNIST/{file_name}")

    print("Done")


def convert_file():
    train_image = f"{save_path}/MNIST/{file_names[0]}"
    train_label = f"{save_path}/MNIST/{file_names[1]}"
        

save_path = "/Users/jun/Downloads"
base_url = "http://yann.lecun.com/exdb/mnist"
file_names = ('train-images-idx3-ubyte.gz',
              'train-labels-idx1-ubyte.gz',
              't10k-images-idx3-ubyte.gz',
              't10k-labels-idx1-ubyte.gz')

save_file = f"{save_path}/mnist.pkl"

download_file()