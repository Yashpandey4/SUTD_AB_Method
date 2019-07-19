
import tensorflow as tf
import tensorflowvisu
import math
import mnistdata
import numpy as np


#Reproducibility
tf.set_random_seed(0)
np.random.seed(0)

mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])  #Input

Y_ = tf.placeholder(tf.float32, [None, 10])  #Correct answers

# variable learning rate
lr = tf.placeholder(tf.float32)
      
# step for variable learning rate
step = tf.placeholder(tf.int32)

ptrans = 0.5    # Probability of transferring weights from A to C (1-ptrans is from B to C)

# We can use different pkeep and ptrans for each layer
# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 200
M = 100
N = 60
O = 30

# Setup parameters of Network A
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
# truncated_normal : The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked. Mean 0 by default.
W1a = tf.get_variable("W1a", shape=[784, L],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)) # 784 = 28 * 28
B1a = tf.Variable(tf.ones([L])/10)
W2a = tf.get_variable("W2a", shape=[L, M],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
B2a = tf.Variable(tf.ones([M])/10)
W3a = tf.get_variable("W3a", shape=[M, N],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
B3a = tf.Variable(tf.ones([N])/10)
W4a = tf.get_variable("W4a", shape=[N, O],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
B4a = tf.Variable(tf.ones([O])/10)
W5a = tf.get_variable("W5a", shape=[O, 10],
           initializer=tf.contrib.layers.xavier_initializer())
B5a = tf.Variable(tf.zeros([10]))    
    
#Biases are common for networks A,B,C


# Setup parameters of Network B
W1b = tf.get_variable("W1b", shape=[784, L],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)) # 784 = 28 * 28
B1b = tf.Variable(tf.ones([L])/10)
W2b = tf.get_variable("W2b", shape=[L, M],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
B2b = tf.Variable(tf.ones([M])/10)
W3b = tf.get_variable("W3b", shape=[M, N],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
B3b = tf.Variable(tf.ones([N])/10)
W4b = tf.get_variable("W4b", shape=[N, O],
           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
B4b = tf.Variable(tf.ones([O])/10)
W5b = tf.get_variable("W5b", shape=[O, 10],
           initializer=tf.contrib.layers.xavier_initializer())
B5b = tf.Variable(tf.zeros([10]))

# Transfer weights and biases from A and B to C
# Loop transfer -> Train C -> Update A,B

def genRandMat(M,N,pt):
    return (np.random.choice([0., 1.], size=(M,N), p=[1-pt, pt]))

trans_arr1 = tf.Variable(genRandMat(784,L,ptrans), dtype=tf.float32)  # 784 = 28 * 28
trans_arr2 = tf.Variable(genRandMat(L,M,ptrans), dtype=tf.float32)
trans_arr3 = tf.Variable(genRandMat(M,N,ptrans), dtype=tf.float32)
trans_arr4 = tf.Variable(genRandMat(N,O,ptrans), dtype=tf.float32)
trans_arr5 = tf.Variable(genRandMat(O,10,ptrans), dtype=tf.float32)

# Setup parameters of Network C
W1c = tf.Variable(tf.add(tf.multiply(W1a, trans_arr1),tf.multiply(W1b,tf.subtract(tf.ones([784, L]),trans_arr1))))  # 784 = 28 * 28
B1c = tf.Variable(tf.ones([L])/10)
W2c = tf.Variable(tf.add(tf.multiply(W2a, trans_arr2),tf.multiply(W2b,tf.subtract(tf.ones([L,M]),trans_arr2))))
B2c = tf.Variable(tf.ones([M])/10)
W3c = tf.Variable(tf.add(tf.multiply(W3a, trans_arr3),tf.multiply(W3b,tf.subtract(tf.ones([M,N]),trans_arr3))))
B3c = tf.Variable(tf.ones([N])/10)
W4c = tf.Variable(tf.add(tf.multiply(W4a, trans_arr4),tf.multiply(W4b,tf.subtract(tf.ones([N,O]),trans_arr4))))
B4c = tf.Variable(tf.ones([O])/10)
W5c = tf.Variable(tf.add(tf.multiply(W5a, trans_arr5),tf.multiply(W5b,tf.subtract(tf.ones([O,10]),trans_arr5))))
B5c = tf.Variable(tf.zeros([10]))

# The model, with dropout at each layer
XX = tf.reshape(X, [-1, 28*28])

Y1 = tf.nn.relu(tf.matmul(XX, W1c) + B1c)

Y2 = tf.nn.relu(tf.matmul(Y1, W2c) + B2c)

Y3 = tf.nn.relu(tf.matmul(Y2, W3c) + B3c)

Y4 = tf.nn.relu(tf.matmul(Y3, W4c) + B4c)

Ylogits = tf.matmul(Y4, W5c) + B5c
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step,
# the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


# Test on trained Models A, B and their hybrid

# Form the Neural Net for A
XXa = tf.reshape(X, [-1, 28*28])

Y1a = tf.nn.relu(tf.matmul(XXa, W1a) + B1a)

Y2a = tf.nn.relu(tf.matmul(Y1a, W2a) + B2a)

Y3a = tf.nn.relu(tf.matmul(Y2a, W3a) + B3a)

Y4a = tf.nn.relu(tf.matmul(Y3a, W4a) + B4a)

Ylogits_a = tf.matmul(Y4a, W5a) + B5a
Ya = tf.nn.softmax(Ylogits_a)

cross_entropy_a = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits_a, labels=Y_)
cross_entropy_a = tf.reduce_mean(cross_entropy_a)*100

correct_prediction_a = tf.equal(tf.argmax(Ya, 1), tf.argmax(Y_, 1))
accuracy_a = tf.reduce_mean(tf.cast(correct_prediction_a, tf.float32))

# Form the Neural Net for B
XXb = tf.reshape(X, [-1, 28*28])

Y1b = tf.nn.relu(tf.matmul(XXb, W1b) + B1b)

Y2b = tf.nn.relu(tf.matmul(Y1b, W2b) + B2b)

Y3b = tf.nn.relu(tf.matmul(Y2b, W3b) + B3b)

Y4b = tf.nn.relu(tf.matmul(Y3b, W4b) + B4b)

Ylogits_b = tf.matmul(Y4b, W5b) + B5b
Yb = tf.nn.softmax(Ylogits_b)

cross_entropy_b = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits_b, labels=Y_)
cross_entropy_b = tf.reduce_mean(cross_entropy_b)*100

correct_prediction_b = tf.equal(tf.argmax(Yb, 1), tf.argmax(Y_, 1))
accuracy_b = tf.reduce_mean(tf.cast(correct_prediction_b, tf.float32))

# Form Neural Net for a hybrid of A, B (subscript h)
W1h=ptrans*W1a+(1-ptrans)*W1b
W2h=ptrans*W2a+(1-ptrans)*W2b
W3h=ptrans*W3a+(1-ptrans)*W3b
W4h=ptrans*W4a+(1-ptrans)*W4b
W5h=ptrans*W5a+(1-ptrans)*W5b
B1h=ptrans*B1a+(1-ptrans)*B1b
B2h=ptrans*B2a+(1-ptrans)*B2b
B3h=ptrans*B3a+(1-ptrans)*B3b
B4h=ptrans*B4a+(1-ptrans)*B4b
B5h=ptrans*B5a+(1-ptrans)*B5b

XXh = tf.reshape(X, [-1, 28*28])

Y1h = tf.nn.relu(tf.matmul(XXh, W1h) + B1h)

Y2h = tf.nn.relu(tf.matmul(Y1h, W2h) + B2h)

Y3h = tf.nn.relu(tf.matmul(Y2h, W3h) + B3h)

Y4h = tf.nn.relu(tf.matmul(Y3h, W4h) + B4h)

Ylogits_h = tf.matmul(Y4h, W5h) + B5h
Yh = tf.nn.softmax(Ylogits_h)

cross_entropy_h = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits_h, labels=Y_)
cross_entropy_h = tf.reduce_mean(cross_entropy_h)*100

correct_prediction_h = tf.equal(tf.argmax(Yh, 1), tf.argmax(Y_, 1))
accuracy_h = tf.reduce_mean(tf.cast(correct_prediction_h, tf.float32))


# *************** Core Logic *************
ass1=(W1a.assign(tf.add(tf.multiply(W1c, trans_arr1),tf.multiply(W1a,tf.subtract(tf.ones([784, L]),trans_arr1)))))
ass2=(W1b.assign(tf.add(tf.multiply(W1b, trans_arr1),tf.multiply(W1c,tf.subtract(tf.ones([784, L]),trans_arr1)))))
ass3=(W2a.assign(tf.add(tf.multiply(W2c, trans_arr2),tf.multiply(W2a,tf.subtract(tf.ones([L,M]),trans_arr2)))))
ass4=(W2b.assign(tf.add(tf.multiply(W2b, trans_arr2),tf.multiply(W2c,tf.subtract(tf.ones([L,M]),trans_arr2)))))
ass5=(W3a.assign(tf.add(tf.multiply(W3c, trans_arr3),tf.multiply(W3a,tf.subtract(tf.ones([M,N]),trans_arr3)))))
ass6=(W3b.assign(tf.add(tf.multiply(W3b, trans_arr3),tf.multiply(W3c,tf.subtract(tf.ones([M,N]),trans_arr3)))))
ass7=(W4a.assign(tf.add(tf.multiply(W4c, trans_arr4),tf.multiply(W4a,tf.subtract(tf.ones([N,O]),trans_arr4)))))
ass8=(W4b.assign(tf.add(tf.multiply(W4b, trans_arr4),tf.multiply(W4c,tf.subtract(tf.ones([N,O]),trans_arr4)))))
ass9=(W5a.assign(tf.add(tf.multiply(W5c, trans_arr5),tf.multiply(W5a,tf.subtract(tf.ones([O,10]),trans_arr5)))))
ass0=(W5b.assign(tf.add(tf.multiply(W5b, trans_arr5),tf.multiply(W5c,tf.subtract(tf.ones([O,10]),trans_arr5)))))

a1=W1c.assign(tf.add(tf.multiply(W1a, trans_arr1),tf.multiply(W1b,tf.subtract(tf.ones([784, L]),trans_arr1))))
a2=W2c.assign(tf.add(tf.multiply(W2a, trans_arr2),tf.multiply(W2b,tf.subtract(tf.ones([L,M]),trans_arr2))))
a3=W3c.assign(tf.add(tf.multiply(W3a, trans_arr3),tf.multiply(W3b,tf.subtract(tf.ones([M,N]),trans_arr3))))
a4=W4c.assign(tf.add(tf.multiply(W4a, trans_arr4),tf.multiply(W4b,tf.subtract(tf.ones([N,O]),trans_arr4))))
a5=W5c.assign(tf.add(tf.multiply(W5a, trans_arr5),tf.multiply(W5b,tf.subtract(tf.ones([O,10]),trans_arr5))))


# init
sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(1)

    # compute training values for visualisation
    # if update_train_data:
    #     a, c, l = sess.run([accuracy, cross_entropy, lr],
    #                             feed_dict={X: batch_X, Y_: batch_Y, step: i})
    #     print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(l) + ")")

    # compute test values for visualisation
    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        print(str(i) + ": ********* epoch " + str(i//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, step: i})



epoch=5
for i in range((epoch*60000)+1):

    # Train C
    training_step(i, i % 10000 == 0, i % 2000 == 0)

    # Update weights of A,B based on new weights of C
    sess.run([ass1,ass2,ass3,ass4,ass5,ass6,ass7,ass8,ass9,ass0])

    # Update transfer arrays
    trans_arr1.load(genRandMat(784,L,ptrans),sess)  # 784 = 28 * 28
    trans_arr2.load(genRandMat(L,M,ptrans),sess)
    trans_arr3.load(genRandMat(M,N,ptrans),sess)
    trans_arr4.load(genRandMat(N,O,ptrans),sess)
    trans_arr5.load(genRandMat(O,10,ptrans),sess)

    # Update weights of C based on weights A and B
    sess.run([a1,a2,a3,a4,a5])



print("\n ************* FINAL MODEL TESTING **************")

a_a, c_a = sess.run([accuracy_a, cross_entropy_a],
                    feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
print(" ********* FINAL MODEL A : test accuracy:" + str(a_a) + " test loss: " + str(c_a)+"  *********")

a_b, c_b = sess.run([accuracy_b, cross_entropy_b],
                    feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
print(" ********* FINAL MODEL B : test accuracy:" + str(a_b) + " test loss: " + str(c_b)+"  *********")

a_h, c_h = sess.run([accuracy_h, cross_entropy_h],
                    feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
print(" ********* FINAL MODEL H : test accuracy:" + str(a_h) + " test loss: " + str(c_h)+"  *********")
