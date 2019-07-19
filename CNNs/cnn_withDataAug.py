# https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/

import tarfile
import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import cv2

tf.set_random_seed(0)
np.random.seed(0)

cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)
    
    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch #{}:'.format(batch_id))
    print('# of Samples: {}\n'.format(len(features)))
    
    label_names = load_label_names()
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))
    
    sample_image = features[sample_id]
    sample_label = labels[sample_id]
    
    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    
    plt.imshow(sample_image)

def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 10))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded

def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))


def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        
        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        # preprocess the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch
        _preprocess_and_save(normalize, one_hot_encode,
                             features[:-index_of_validation], labels[:-index_of_validation], 
                             'preprocess_batch_' + str(batch_i) + '.p')

        # unlike the training dataset, validation dataset will be added through all batch dataset
        # - take 10% of the whold dataset of the batch
        # - add them into a list of
        #   - valid_features
        #   - valid_labels
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(valid_features), np.array(valid_labels),
                         'preprocess_validation.p')

    # load the test dataset
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all testing data
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(test_features), np.array(test_labels),
                         'preprocess_training.p')

def conv_net(x, keep_prob, conv1_filter ,conv2_filter ,conv3_filter ,conv4_filter):
    

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_pool, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv2_bn = tf.layers.batch_normalization(conv2_pool)

    # 5, 6
    conv3 = tf.nn.conv2d(conv2_pool, conv3_filter, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv3_bn = tf.layers.batch_normalization(conv3_pool)

    # 7, 8
    conv4 = tf.nn.conv2d(conv3_pool, conv4_filter, strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv4_bn = tf.layers.batch_normalization(conv4_pool)

    # 9
    flat = tf.contrib.layers.flatten(conv4_pool)

    return flat

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer, 
                feed_dict={
                    x: feature_batch,
                    y: label_batch,
                    keep_prob: keep_probability
                })

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    loss = sess.run(cost, 
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: 1.
                    })
    valid_acc = sess.run(accuracy, 
                         feed_dict={
                             x: valid_features,
                             y: valid_labels,
                             keep_prob: 1.
                         })
    
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


# ========================================================== #
# ├─ _random_crop() 
# ├─ _random_flip_leftright()
# ├─ data_augmentation()
# └─ color_preprocessing()
# ========================================================== #

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    
    if padding:
        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                        mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                    nw:nw + crop_shape[1]]
    return new_batch

def shift_left(batch):
    for k in range(len(batch)):
        img=batch[k]
        if bool(random.getrandbits(1)):
            # Shifting Left
            for i in range(32, 1, -1):
                for j in range(32):
                    if (i < 32-5):
                        img[j][i] = img[j][i-5]
                    elif (i < 32-5):
                        img[j][i] = 0
        batch[k]=img
    return batch

def shift_right(batch):
    for k in range(len(batch)):
        img=batch[k]
        if bool(random.getrandbits(1)):
            # Shifting Right
            for j in range(32):
                for i in range(32):
                    if (i < 32-5):
                        img[j][i] = img[j][i+5]
        batch[k]=img
    return batch

def shift_up(batch):
    for k in range(len(batch)):
        img=batch[k]
        if bool(random.getrandbits(1)):
             # Shifting Up
            for j in range(32):
                for i in range(32):
                    if (j < 32 - 5 and j > 5):
                        img[j][i] = img[j+5][i]
                    else:
                        img[j][i] = 0
        batch[k]=img
    return batch

def shift_down(batch):
    for k in range(len(batch)):
        img=batch[k]
        if bool(random.getrandbits(1)):
            #Shifting Down
            for j in range(32, 1, -1):
                for i in range(32):
                    if (j < 25 and j > 5):
                        img[j][i] = img[j-5][i]
        batch[k]=img
    return batch

def noise(batch):
    for l in range(len(batch)):
        img=batch[l]
        if bool(random.getrandbits(1)):
            # ADDING NOISE
            noise = np.random.randint(5, size = (32,32,3), dtype = 'uint8')

            for i in range(32):
                for j in range(32):
                    for k in range(3):
                        if (img[i][j][k] != 255):
                            img[i][j][k] += noise[i][j][k]
        batch[l]=img
    return batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test

def central_scale_images(batch,sess,scales):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            # Various settings needed for Tensorflow operation
            boxes = np.zeros((len(scales), 4), dtype = np.float32)
            for index, scale in enumerate(scales):
                x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
                x2 = y2 = 0.5 + 0.5 * scale
                boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
            box_ind = np.zeros((len(scales)), dtype = np.int32)
            crop_size = np.array([32, 32], dtype = np.int32)
            
            # Define Tensorflow operation for all scales but only one base image at a time
        
            batch_img = np.expand_dims(batch[i], axis = 0)
            scaled_imgs = sess.run(tf_img_scale, feed_dict = {X1: batch_img, boxes1: boxes, box_ind1: box_ind, crop_size1: crop_size})
            k=np.random.randint(3)
            batch[i] = scaled_imgs[k]
    return batch

def add_salt_pepper_noise(batch):
    # Need to produce a copy as to not modify the original image
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            X_imgs = batch[i]
            salt_vs_pepper = 0.2
            amount = 0.004
            num_salt = np.ceil(amount * len(batch) * salt_vs_pepper)
            num_pepper = np.ceil(amount * len(batch) * (1.0 - salt_vs_pepper))
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_imgs.shape]
            X_imgs[coords[0], coords[1], :] = 1

            # Add Pepper noise
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_imgs.shape]
            X_imgs[coords[0], coords[1], :] = 0
            batch[i] = X_imgs
    return batch

def add_gaussian_noise(batch):
    row, col, _ = batch[0].shape
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            gaussian_noise_imgs = []
            # Gaussian distribution parameters
            X_imgs=batch[i]
            gaussian = np.random.random((row, col, 1)).astype(np.float32)
            gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
            temp=np.array(0.25 * gaussian, dtype = np.float32)
            gaussian_img = (X_imgs * 0.75 + temp * 0.25)
            gaussian_noise_imgs.append(gaussian_img)
            gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
            batch[i] = X_imgs
    return batch

X = tf.placeholder(tf.float32, shape = (32,32, 3))
X1 = tf.placeholder(tf.float32, shape = (1,32,32, 3))
k = tf.placeholder(tf.int32)
radian = tf.placeholder(tf.float32, shape = (1))
boxes1=tf.placeholder(tf.float32, shape = (3,4))
box_ind1 = tf.placeholder(tf.int32, shape = (3))
crop_size1 = tf.placeholder(tf.int32, shape = [2])
tf_img90 = tf.image.rot90(X, k = k)
tf_img = tf.contrib.image.rotate(X, radian)
tf_img_scale = tf.image.crop_and_resize(X1, boxes1, box_ind1, crop_size1)

def rotate90(batch,sess):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            j = np.random.randint(3)
            rotated_img = sess.run(tf_img90, feed_dict = {X: batch[i], k: j + 1})
            batch[i]=rotated_img
    return batch

def rotate(batch,sess):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            degrees_angle = np.random.choice(np.arange(-15.,15.,2.5))
            radian_value = [degrees_angle * 3.14 / 180.]  # Convert to radian
            rotated_img = sess.run(tf_img, feed_dict = {X: batch[i], radian: radian_value})
            batch[i]=rotated_img
    return batch

# https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9

def data_augmentation(batch,sess):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32,32], 4)
    # batch = shift_left(batch)
    # batch = shift_right(batch)
    # batch = shift_up(batch)
    # batch = shift_down(batch)
    # batch = noise(batch)
    # batch = rotate90(batch,sess)
    batch = rotate(batch,sess)
    # batch = central_scale_images(batch,sess,[0.90, 0.75, 0.60])
    # batch = add_salt_pepper_noise(batch)
    # batch = add_gaussian_noise(batch)
    return batch

# End of helper methods. Code begins here.

# Download the dataset (if not exist yet)
# if not isfile('cifar-10-python.tar.gz'):
#     with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
#         urlretrieve(
#             'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
#             'cifar-10-python.tar.gz',
#             pbar.hook)

# if not isdir(cifar10_dataset_folder_path):
#     with tarfile.open('cifar-10-python.tar.gz') as tar:
#         tar.extractall()
#         tar.close()

# Explore the dataset
batch_id = 3
sample_id = 7000
# display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

# Preprocess all the data and save it
# preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

# load the saved dataset
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

# Hyper parameters
epochs = 50
batch_size = 128
keep_probability = 1
learning_rate = 0.001       # random but researched - 0.0005, 0.001, 0.00146 performed best
ptrans = 0.5

# Remove previous weights, bias, inputs, etc..
# tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

def genRandMatConv(M,N,O,P,pt):
    return (np.random.choice([0., 1.], size=(M,N,O,P), p=[1-pt, pt]))

trans_arr1Conv = tf.Variable(genRandMatConv(3, 3, 3, 64 ,ptrans), dtype=tf.float32) 
trans_arr2Conv = tf.Variable(genRandMatConv(3, 3, 64, 128,ptrans), dtype=tf.float32)
trans_arr3Conv = tf.Variable(genRandMatConv(5, 5, 128, 256, ptrans), dtype=tf.float32)
trans_arr4Conv = tf.Variable(genRandMatConv(5, 5, 256, 512, ptrans), dtype=tf.float32)

def genRandMat(M,N,pt):
    return (np.random.choice([0., 1.], size=(M,N), p=[1-pt, pt]))

L = 128
M = 256
N = 512

trans_arr1 = tf.Variable(genRandMat(2048,L,ptrans), dtype=tf.float32) 
trans_arr2 = tf.Variable(genRandMat(L,M,ptrans), dtype=tf.float32)
trans_arr3 = tf.Variable(genRandMat(M,N,ptrans), dtype=tf.float32)
trans_arr4 = tf.Variable(genRandMat(N,10,ptrans), dtype=tf.float32)

# Define for C
conv1_filter_c = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
conv2_filter_c = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
conv3_filter_c = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
conv4_filter_c = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

# Define for H
conv1_filter_h = tf.Variable(conv1_filter_c)
conv2_filter_h = tf.Variable(conv2_filter_c)
conv3_filter_h = tf.Variable(conv3_filter_c)
conv4_filter_h = tf.Variable(conv4_filter_c)

# Define for a
conv1_filter_a = tf.Variable(conv1_filter_c)
conv2_filter_a = tf.Variable(conv2_filter_c)
conv3_filter_a = tf.Variable(conv3_filter_c)
conv4_filter_a = tf.Variable(conv4_filter_c)

# Define for b
conv1_filter_b = tf.Variable(conv1_filter_c)
conv2_filter_b = tf.Variable(conv2_filter_c)
conv3_filter_b = tf.Variable(conv3_filter_c)
conv4_filter_b = tf.Variable(conv4_filter_c)

# *****************Build model for A**************************
flat_a = conv_net(x, keep_prob, conv1_filter_a ,conv2_filter_a ,conv3_filter_a ,conv4_filter_a)

W1a = tf.get_variable("W1a", shape=[2048, L],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)) # 784 = 28 * 28
B1a = tf.Variable(tf.ones([L])/10)
W2a = tf.get_variable("W2a", shape=[L, M],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
B2a = tf.Variable(tf.ones([M])/10)
W3a = tf.get_variable("W3a", shape=[M, N],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
B3a = tf.Variable(tf.ones([N])/10)
W4a = tf.get_variable("W4a", shape=[N, 10],
           initializer=tf.contrib.layers.xavier_initializer())
B4a = tf.Variable(tf.zeros([10])) 


Y1_a = tf.nn.relu(tf.matmul(flat_a, W1a) + B1a)

Y2_a = tf.nn.relu(tf.matmul(Y1_a, W2a) + B2a)

Y3_a = tf.nn.relu(tf.matmul(Y2_a, W3a) + B3a)

logits_a = (tf.matmul(Y3_a, W4a) + B4a)

model_a = tf.identity(logits_a, name='logits_a') # Name logits Tensor, so that can be loaded from disk after training

# Loss and Optimizer
cost_a = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_a, labels=y))
optimizer_a = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_a)

# Accuracy
correct_pred_a = tf.equal(tf.argmax(logits_a, 1), tf.argmax(y, 1))
accuracy_a = tf.reduce_mean(tf.cast(correct_pred_a, tf.float32), name='accuracy_a')

# *****************Build model for B**************************
flat_b = conv_net(x, keep_prob, conv1_filter_b ,conv2_filter_b ,conv3_filter_b ,conv4_filter_b)

W1b = tf.get_variable("W1b", shape=[2048, L],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)) # 784 = 28 * 28
B1b = tf.Variable(tf.ones([L])/10)
W2b = tf.get_variable("W2b", shape=[L, M],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
B2b = tf.Variable(tf.ones([M])/10)
W3b = tf.get_variable("W3b", shape=[M, N],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
B3b = tf.Variable(tf.ones([N])/10)
W4b = tf.get_variable("W4b", shape=[N, 10],
           initializer=tf.contrib.layers.xavier_initializer())
B4b = tf.Variable(tf.zeros([10])) 

Y1_b = tf.nn.relu(tf.matmul(flat_b, W1b) + B1b)

Y2_b = tf.nn.relu(tf.matmul(Y1_b, W2b) + B2b)

Y3_b = tf.nn.relu(tf.matmul(Y2_b, W3b) + B3b)

logits_b = (tf.matmul(Y3_b, W4b) + B4b)

model_b = tf.identity(logits_b, name='logits_b') # Name logits Tensor, so that can be loaded from disk after training

# Loss and Optimizer
cost_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_b, labels=y))
optimizer_b = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_b)

# Accuracy
correct_pred_b = tf.equal(tf.argmax(logits_b, 1), tf.argmax(y, 1))
accuracy_b = tf.reduce_mean(tf.cast(correct_pred_b, tf.float32), name='accuracy_b')

# *****************Build model for C**************************

flat_c = conv_net(x, keep_prob, conv1_filter_c ,conv2_filter_c ,conv3_filter_c ,conv4_filter_c)

W1c = tf.Variable(tf.add(tf.multiply(W1a, trans_arr1),tf.multiply(W1b,tf.subtract(tf.ones([2048,L]),trans_arr1))))
B1c = tf.Variable(tf.ones([L])/10)
W2c = tf.Variable(tf.add(tf.multiply(W2a, trans_arr2),tf.multiply(W2b,tf.subtract(tf.ones([L,M]),trans_arr2))))
B2c = tf.Variable(tf.ones([M])/10)
W3c = tf.Variable(tf.add(tf.multiply(W3a, trans_arr3),tf.multiply(W3b,tf.subtract(tf.ones([M,N]),trans_arr3))))
B3c = tf.Variable(tf.ones([N])/10)
W4c = tf.Variable(tf.add(tf.multiply(W4a, trans_arr4),tf.multiply(W4b,tf.subtract(tf.ones([N,10]),trans_arr4))))
B4c = tf.Variable(tf.zeros([10])) 

Y1 = tf.nn.relu(tf.matmul(flat_c, W1c) + B1c)
Y1d = tf.nn.dropout(Y1, keep_prob) * keep_prob
# Y1d = tf.layers.batch_normalization(Y1d)

Y2 = tf.nn.relu(tf.matmul(Y1d, W2c) + B2c)
Y2d = tf.nn.dropout(Y2, keep_prob) * keep_prob
# Y2d = tf.layers.batch_normalization(Y2d)

Y3 = tf.nn.relu(tf.matmul(Y2d, W3c) + B3c)
Y3d = tf.nn.dropout(Y3, keep_prob) * keep_prob
# Y3d = tf.layers.batch_normalization(Y3d)

logits = (tf.matmul(Y3d, W4c) + B4c)

model = tf.identity(logits, name='logits') # Name logits Tensor, so that can be loaded from disk after training

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


# *****************Build model for H**************************
flat_h = conv_net(x, keep_prob, conv1_filter_h ,conv2_filter_h ,conv3_filter_h ,conv4_filter_h)

W1h=(ptrans*W1a+(1-ptrans)*W1b)
W2h=(ptrans*W2a+(1-ptrans)*W2b)
W3h=(ptrans*W3a+(1-ptrans)*W3b)
W4h=(ptrans*W4a+(1-ptrans)*W4b)
B1h=(ptrans*B1a+(1-ptrans)*B1b)
B2h=(ptrans*B2a+(1-ptrans)*B2b)
B3h=(ptrans*B3a+(1-ptrans)*B3b)
B4h=(ptrans*B4a+(1-ptrans)*B4b)

Y1_h = tf.nn.relu(tf.matmul(flat_h, W1h) + B1h)

Y2_h = tf.nn.relu(tf.matmul(Y1_h, W2h) + B2h)

Y3_h = tf.nn.relu(tf.matmul(Y2_h, W3h) + B3h)

logits_h = (tf.matmul(Y3_h, W4h) + B4h)

model_h = tf.identity(logits_h, name='logits_h') # Name logits Tensor, so that can be loaded from disk after training

# Loss and Optimizer
cost_h = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_h, labels=y))
optimizer_h = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_h)

# Accuracy
correct_pred_h = tf.equal(tf.argmax(logits_h, 1), tf.argmax(y, 1))
accuracy_h = tf.reduce_mean(tf.cast(correct_pred_h, tf.float32), name='accuracy_h')

# ********************* CORE LOGIC ***************************

#Update A,B Fully Connected Weights based on trained parameters of C
w_ass1=(W1a.assign(tf.add(tf.multiply(W1c, trans_arr1),tf.multiply(W1a,tf.subtract(tf.ones([2048, L]),trans_arr1)))))
w_ass2=(W1b.assign(tf.add(tf.multiply(W1b, trans_arr1),tf.multiply(W1c,tf.subtract(tf.ones([2048, L]),trans_arr1)))))
w_ass3=(W2a.assign(tf.add(tf.multiply(W2c, trans_arr2),tf.multiply(W2a,tf.subtract(tf.ones([L,M]),trans_arr2)))))
w_ass4=(W2b.assign(tf.add(tf.multiply(W2b, trans_arr2),tf.multiply(W2c,tf.subtract(tf.ones([L,M]),trans_arr2)))))
w_ass5=(W3a.assign(tf.add(tf.multiply(W3c, trans_arr3),tf.multiply(W3a,tf.subtract(tf.ones([M,N]),trans_arr3)))))
w_ass6=(W3b.assign(tf.add(tf.multiply(W3b, trans_arr3),tf.multiply(W3c,tf.subtract(tf.ones([M,N]),trans_arr3)))))
w_ass7=(W4a.assign(tf.add(tf.multiply(W4c, trans_arr4),tf.multiply(W4a,tf.subtract(tf.ones([N,10]),trans_arr4)))))
w_ass8=(W4b.assign(tf.add(tf.multiply(W4b, trans_arr4),tf.multiply(W4c,tf.subtract(tf.ones([N,10]),trans_arr4)))))

#Update C Fully Connected Weights based parameters of A and B
w_a1=W1c.assign(tf.add(tf.multiply(W1a, trans_arr1),tf.multiply(W1b,tf.subtract(tf.ones([2048,L]),trans_arr1))))
w_a2=W2c.assign(tf.add(tf.multiply(W2a, trans_arr2),tf.multiply(W2b,tf.subtract(tf.ones([L,M]),trans_arr2))))
w_a3=W3c.assign(tf.add(tf.multiply(W3a, trans_arr3),tf.multiply(W3b,tf.subtract(tf.ones([M,N]),trans_arr3))))
w_a4=W4c.assign(tf.add(tf.multiply(W4a, trans_arr4),tf.multiply(W4b,tf.subtract(tf.ones([N,10]),trans_arr4))))

# new assign - assign A and B, filters of C post training
ass1=conv1_filter_a.assign(conv1_filter_c)
ass2=conv1_filter_b.assign(conv1_filter_c)
ass3=conv2_filter_a.assign(conv2_filter_c)
ass4=conv2_filter_b.assign(conv2_filter_c)
ass5=conv3_filter_a.assign(conv3_filter_c)
ass6=conv3_filter_b.assign(conv3_filter_c)
ass7=conv4_filter_a.assign(conv4_filter_c)
ass8=conv4_filter_b.assign(conv4_filter_c)
ass9=conv1_filter_h.assign(conv1_filter_c)
ass10=conv2_filter_h.assign(conv2_filter_c)
ass11=conv3_filter_h.assign(conv3_filter_c)
ass12=conv4_filter_h.assign(conv4_filter_c)

# Training Phase
save_model_path = './image_classification'

saver = tf.train.Saver()

init = tf.global_variables_initializer()

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(init)

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                batch_features = data_augmentation(batch_features,sess)
                #Train C
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)

                # Update weights of A,B based on new weights of C
                sess.run([w_ass1,w_ass2,w_ass3,w_ass4,w_ass5,w_ass6,w_ass7,w_ass8])


                # Update transfer arrays
                trans_arr1.load(genRandMat(2048,L,ptrans))
                trans_arr2.load(genRandMat(L,M,ptrans))  # 784 = 28 * 28
                trans_arr3.load(genRandMat(M,N,ptrans))
                trans_arr4.load(genRandMat(N,10,ptrans))

                # Update weights of C based on weights A and B
                sess.run([w_a1,w_a2,w_a3,w_a4])

            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
    
    idx=[0]

    sess.run([ass1,ass2,ass3,ass4,ass5,ass6,ass7,ass8,ass9,ass10,ass11,ass12])

    # print(tf.gather(W1c,idx).eval())
    # print(tf.gather(trans_arr1,idx).eval())
    # print(tf.gather(W1a,idx).eval())
    # print()
    # print(tf.gather(W1b,idx).eval())

    print("\n ************* FINAL MODEL TESTING **************")

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    test_batch_acc_total = 0
    test_batch_count = 0
    for train_feature_batch, train_label_batch in batch_features_labels(test_features, test_labels, 64):
            test_batch_acc_total += sess.run(
                accuracy,
                feed_dict={x: train_feature_batch, y: train_label_batch, keep_prob: 1.0})
            test_batch_count += 1

    print('********* FINAL MODEL C : Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))
    
    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    test_batch_acc_total = 0
    test_batch_count = 0
    for train_feature_batch, train_label_batch in batch_features_labels(test_features, test_labels, 64):
            test_batch_acc_total += sess.run(
                accuracy_a,
                feed_dict={x: train_feature_batch, y: train_label_batch, keep_prob: 1.0})
            test_batch_count += 1

    print('********* FINAL MODEL A : Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

    test_batch_acc_total = 0
    test_batch_count = 0
    for train_feature_batch, train_label_batch in batch_features_labels(test_features, test_labels, 64):
            test_batch_acc_total += sess.run(
                accuracy_b,
                feed_dict={x: train_feature_batch, y: train_label_batch, keep_prob: 1.0})
            test_batch_count += 1

    print('********* FINAL MODEL B : Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))
    
    test_batch_acc_total = 0
    test_batch_count = 0
    for train_feature_batch, train_label_batch in batch_features_labels(test_features, test_labels, 64):
            test_batch_acc_total += sess.run(
                accuracy_h,
                feed_dict={x: train_feature_batch, y: train_label_batch, keep_prob: 1.0})
            test_batch_count += 1

    print('********* FINAL MODEL H : Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))


    # Save Model
    save_path = saver.save(sess, save_model_path)