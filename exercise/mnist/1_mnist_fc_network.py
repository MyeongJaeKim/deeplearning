import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

# 트레이닝 데이터 다운로드 (from MNIST Web Site)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 트레이닝 파라미터
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# Placeholders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# Fully Connected Layers

# Layer 1
W1 = tf.get_variable("W1", shape=[784, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# Layer 2
W2 = tf.get_variable("W2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# Layer 3
W3 = tf.get_variable("W3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# Layer 4
W4 = tf.get_variable("W4", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# Output layer
W5 = tf.get_variable("W5", shape=[512, 10])
b5 = tf.Variable(tf.random_normal([10]))

# 트레이닝 Formulation
hypothesis = tf.matmul(L4, W5) + b5
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Tensorflow Session 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 트레인
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        sampling_x, sampling_y = mnist.train.next_batch(batch_size)
        feed_dict = {X: sampling_x, Y: sampling_y, keep_prob: 0.7}
        currentcost, _ = sess.run([cost, optimizer], feed_dict=feed_dict)

        # 평균 = (x1 + x2 + ... + xn) / N = x1 / N + x2 / N + ... + xn / N
        avg_cost += currentcost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# 모델 평가
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))









