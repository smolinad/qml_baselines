import tensorflow as tf
import numpy as np
import collections

"""
Module for MNIST Digits Dataset preprocessing.
https://www.tensorflow.org/quantum/tutorials/mnist

Python 3.10.11
"""

def filter_by_classes(x, y, classes=[3,6]):
    """
    Function that filters the MNIST Digits Dataset and returns samples on 'classes'.
    Parameters:
        x: Sample images.
        y: Sample labels.
        classes: List of classes to filter.
    Returns:
        x: x filtered by 'classes'.
        y: x filtered by 'classes'.
    """
    if not all(np.isin(classes, range(0, 10))):
        return ValueError("Classes must be a list of digits (0-9).")
    x, y = x[np.isin(y, classes)], y[np.isin(y, classes)]
    if len(classes)==2:
        return x, y==classes[-1]
    else:
        return x, y

def remove_contradicting(xs, ys):

    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
       orig_x[tuple(x.flatten())] = x
       mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for flatten_x in mapping:
      x = orig_x[flatten_x]
      labels = mapping[flatten_x]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(next(iter(labels)))
      else:
          # Throw out images that match more than one label.
          pass

    num_uniq_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num_uniq_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)

    print("Number of unique images:", len(mapping.values()))
    print("Number of unique 3s: ", num_uniq_3)
    print("Number of unique 6s: ", num_uniq_6)
    print("Number of unique contradicting labels (both 3 and 6): ", num_uniq_both)
    print()
    print("Initial number of images: ", len(xs))
    print("Remaining non-contradicting unique images: ", len(new_x))

    return np.array(new_x), np.array(new_y)

def preprocess_mnist_digits(classes=[3,6]):
    """"
    Function that downloads the MNIST Digits dataset with TensorFlow and performs the following tasks:
        1. Normalizes pixel values from (0, 255) to (0, 1).
        2. By default, returns only 2 classes of digits for classification (this can be deactivated or modified by the 'classes' parameter).
        3. Resizes samples to 4x4 images.
        4. Removes samples that belong to multiple classes simultaneously.
        5. Converts images to binary."
    Parameters:
    Returns:
    """

    # Download dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    # Filter to get only '3's and '6's
    x_train, y_train = filter_by_classes(x_train, y_train, classes=classes)
    x_test, y_test = filter_by_classes(x_test, y_test, classes=classes)

    print("Number of filtered training examples:", len(x_train))
    print("Number of filtered test examples:", len(x_test))

    # Resize images to 4x4
    x_train_small = tf.image.resize(x_train, (4,4)).numpy()
    x_test_small = tf.image.resize(x_test, (4,4)).numpy()

    x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)

    THRESHOLD = 0.5

    # Converts non contradicting samples to binary via threshold and converting bool to float.
    x_train_bin = np.array(x_train_nocon > THRESHOLD, dtype=np.float32)
    x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)

    return x_train_bin, np.where(y_train_nocon==False, -1, 1), x_test_bin, np.where(y_test==False, -1, 1)

