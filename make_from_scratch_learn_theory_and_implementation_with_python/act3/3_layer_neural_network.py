# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
# ]
# ///

import os
import gzip
import urllib.request
import pickle
from PIL import Image
import numpy as np


DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FILE = DATASET_DIR + "/mnist.pkl"

KEY_FILE = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}


def _download(file_name):
    file_path = DATASET_DIR + "/" + file_name

    if os.path.exists(file_path):
        print("Error: Not found file. ", file_path)
        return

    print("Downloading " + file_name + " ... ")
    URL_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    headers = { "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0" }
    request = urllib.request.Request(URL_BASE+file_name, headers=headers)
    response = urllib.request.urlopen(request).read()
    with open(file_path, mode='wb') as f:
        f.write(response)
    print("Done")


def download_mnist():
    for v in KEY_FILE.values():
        _download(v)


def _load_label(file_name):
    file_path = DATASET_DIR + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, mode='rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(file_name):
    file_path = DATASET_DIR + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, mode='rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    IMG_SIZE = 784
    data = data.reshape(-1, IMG_SIZE)
    print("Done")

    return data


def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(KEY_FILE['train_img'])
    dataset['train_label'] = _load_label(KEY_FILE['train_label'])
    dataset['test_img'] = _load_img(KEY_FILE['test_img'])
    dataset['test_label'] = _load_label(KEY_FILE['test_label'])

    return dataset


def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(SAVE_FILE, mode='wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(SAVE_FILE):
        init_mnist()

    with open(SAVE_FILE, mode='rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


def show_img(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print("shapes:")
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
print()

print("img:")
img = x_train[0]
