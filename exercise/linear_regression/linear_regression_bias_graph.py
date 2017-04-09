import tensorflow as tf
import matplotlib.pyplot as plt

# 테스트 데이터
x = [1, 2, 3]
y = [1, 2, 3]

# 변수 선언
a = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
b = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

# Y = aX + b
hypothesis = tf.add(tf.multiply(a, X), b)

# Error (=loss) (=cost) 함수
error = tf.reduce_mean(tf.square(hypothesis - Y))

# Tensorflow Train 변수 선언
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(error)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 그래프를 그리기 위한 List
    weights = []
    biases = []

    # 실제 Train
    for step in range(300):
        _, error_val = sess.run([train, error], feed_dict={X: x, Y: y})
        weights.append(sess.run(a))
        biases.append(sess.run(b))
        print("Step {} Error {}".format(step, error_val))

    print("Train was finished. a is {} and b is {}".format(sess.run(a), sess.run(b)))

    print("Let's test real value")
    print("X: 4.87, Y:", sess.run(hypothesis, feed_dict={X: 4.87}))

    print("Drawing graph of a and b")
    plt.figure(1)
    plt.subplot(211)
    plt.plot(weights)
    plt.xlabel("Step")
    plt.ylabel("Weight")
    plt.subplot(212)
    plt.plot(biases)
    plt.xlabel("Step")
    plt.ylabel("Bias")
    plt.show()


