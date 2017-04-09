import tensorflow as tf
import matplotlib.pyplot as plt

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

    # a 값을 그래프 그리기 위한 List
    weights = []

    # 실제 Train
    for step in range(10):
        _, error_val = sess.run([train, error], feed_dict={X: x, Y: y})
        weights.append(sess.run(a))
        print("Step {} Error {}".format(step, error_val))

    print("Train was finished. a is {}".format(sess.run(a)))

    print("Let's test real value")
    print("X: 4.87, Y:", sess.run(hypothesis, feed_dict={X: 4.87}))

    print("Drawing Graph of a")
    plt.plot(weights)
    plt.xlabel("Step")
    plt.ylabel("Weight")
    plt.show()



