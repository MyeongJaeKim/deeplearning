import tensorflow as tf
import os

# 파라미터
CHECK_POINT_DIR = './linear_model'

# Model Saving Directory Existence Check
if not os.path.exists(CHECK_POINT_DIR):
    os.makedirs(CHECK_POINT_DIR)

# 테스트 데이터
x = [1, 2, 3]
y = [1, 2, 3]

# 변수 선언
a = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Y = aX
hypothesis = tf.multiply(a, X)

# Error (=loss) (=cost) 함수
error = tf.reduce_mean(tf.square(hypothesis - Y))

# Tensorflow Train 변수 선언
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(error)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 실제 Train
    for step in range(10):
        _, error_val = sess.run([train, error], feed_dict={X: x, Y: y})
        print("Step {} Error {}".format(step, error_val))

    print("Train was finished. a is {}".format(sess.run(a)))

    # Model 세이브
    print("Saving network...")
    saver = tf.train.Saver()
    tf.add_to_collection('trained_model', hypothesis)
    saver.save(sess, CHECK_POINT_DIR + "/model", global_step=10)

    # Tensorflow Session 초기화
    sess.close()

with tf.Session() as restored:
    restored.run(tf.global_variables_initializer())

    # 모델 복구
    checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
    graph_path = checkpoint.model_checkpoint_path + ".meta"
    graph = tf.train.import_meta_graph(graph_path)
    graph.restore(restored, checkpoint.model_checkpoint_path)

    restored_hypothesis = tf.get_collection('trained_model')[0]

    print("Restoring was successful? Let's see...")

    print("Let's test real value")
    print("X: 4.87, Y:", restored.run(restored_hypothesis, feed_dict={X: 4.87}))


