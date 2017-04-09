import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# 트레이닝 데이터 다운로드 (from MNIST Web Site)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# TensorBoard 데이터 저장 위치
TB_SUMMARY_DIR = './3/tb'

# 훈련된 Model 저장 위치
CHECK_POINT_DIR = './mnist_model'

# 트레이닝 파라미터
learning_rate = 0.001
training_epochs = 20
batch_size = 100

# Placeholders
X = tf.placeholder(tf.float32, [None, 784], name="X")
Y = tf.placeholder(tf.float32, [None, 10], name="Y")

keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# Image input
x_image = tf.reshape(X, [-1, 28, 28, 1])
tf.summary.image('input', x_image, 3)

# Fully Connected Layers

# Layer 1
with tf.variable_scope('layer1') as scope1:
    W1 = tf.get_variable("W1", shape=[784, 512],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([512]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    tf.summary.histogram("X", X)
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("bias", b1)
    tf.summary.histogram("layer", L1)

# Layer 2
with tf.variable_scope('layer2') as scope2:
    W2 = tf.get_variable("W2", shape=[512, 512],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([512]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

    tf.summary.histogram("weights", W2)
    tf.summary.histogram("bias", b2)
    tf.summary.histogram("layer", L2)

# Layer 3
with tf.variable_scope('layer3') as scope3:
    W3 = tf.get_variable("W3", shape=[512, 512],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([512]))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

    tf.summary.histogram("weights", W3)
    tf.summary.histogram("bias", b3)
    tf.summary.histogram("layer", L3)

# Layer 4
with tf.variable_scope('layer4') as scope4:
    W4 = tf.get_variable("W4", shape=[512, 512],
                         initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([512]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

    tf.summary.histogram("weights", W4)
    tf.summary.histogram("bias", b4)
    tf.summary.histogram("layer", L4)

# Output layer
with tf.variable_scope('output') as scope5:
    W5 = tf.get_variable("W5", shape=[512, 10])
    b5 = tf.Variable(tf.random_normal([10]))
    hypothesis = tf.matmul(L4, W5) + b5

    # Originally,
    # hypothesis = tf.nn.softmax(tf.matmul(L4, W5) + b5)
    # Cross Entropy 함수에 통합되어 있음

    tf.summary.histogram("weights", W5)
    tf.summary.histogram("bias", b5)
    tf.summary.histogram("hypothesis", hypothesis)

# 트레이닝 Formulation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Predict 용 Softmax Function
oracle = tf.nn.softmax(hypothesis)

# Model 복구 후 사용하기 위해 Collection 저장
tf.add_to_collection('predictor', oracle)

tf.summary.scalar("cost", cost)

# 마지막 실행한 epoch
last_epoch = tf.Variable(0, name='last_epoch')

# Summary
summary = tf.summary.merge_all()

# Tensorflow Session 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Create summary writer
writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph)
global_step = 0

# Check point 저장
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)

# 트레인
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        sampling_x, sampling_y = mnist.train.next_batch(batch_size)
        feed_dict = {X: sampling_x, Y: sampling_y, keep_prob: 0.7}
        s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
        writer.add_summary(s, global_step=global_step)
        global_step += 1

        # 평균 = (x1 + x2 + ... + xn) / N = x1 / N + x2 / N + ... + xn / N
        avg_cost += sess.run(cost, feed_dict=feed_dict) / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

print("Saving network...")
if not os.path.exists(CHECK_POINT_DIR):
    os.makedirs(CHECK_POINT_DIR)
saver.save(sess, CHECK_POINT_DIR + "/model")

# 모델 평가 (예측 정확도 산출)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
