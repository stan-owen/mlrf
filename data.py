import numpy as np
import os

PATH_TO_RESOURCES = "data/cifar-10-python/"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_batch(batch_filename):
    data = unpickle(batch_filename)
    train_data = data[b'data']
    train_labels = data[b'labels']

    new_train_data = []
    for i in range(len(train_data)):
        new_train_data.append(train_data[i].reshape(3, 32, 32).transpose(1, 2, 0))
    train_data = np.array(new_train_data)

    return train_data, train_labels

def load_data():
    train_batch, train_labels = get_batch(PATH_TO_RESOURCES + "data_batch_1")
    # for i in range(2, 6):
    #     batch, labels = get_batch(PATH_TO_RESOURCES +  "data_batch_" + str(i))
    #     train_batch = np.append(train_batch, batch, axis=0)
    #     train_labels = np.append(train_labels, labels, axis=0)
    test_batch, labels_test = get_batch(PATH_TO_RESOURCES + "test_batch")

    labels_name = [str(l)[2:-1] for l in unpickle(os.path.join(PATH_TO_RESOURCES, "batches.meta"))[b'label_names']]

    return train_batch, train_labels, test_batch, labels_test, labels_name