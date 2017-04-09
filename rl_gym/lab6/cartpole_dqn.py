import rl_gym
import numpy as np
import time
import os
import tensorflow as tf

'''
https://www.youtube.com/watch?v=MF_Wllw9VKk&index=13&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG
'''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

env = rl_gym.make('CartPole-v0')

learning_rate = 1e-1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

X = tf.placeholder(tf.float32, [None, input_size], name="input_x")
W1 = tf.get_variable("W1", shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X, W1)
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

dis = .99
num_episodes = 2000
rList = []

start_time = time.time()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(num_episodes):
        e = 1. / ((i / 10) + 1)
        rAll = 0
        step_count = 0
        s = env.reset()
        done = False

        while not done:
            step_count += 1
            flattened_s = np.reshape(s, [1, input_size])
            Qs = sess.run(Qpred, feed_dict={X: flattened_s})

            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            s1, reward, done, _ = env.step(a)

            if done:
                Qs[0, a] = -100
            else:
                flattened_s1 = np.reshape(s1, [1, input_size])
                Qs1 = sess.run(Qpred, feed_dict={X: flattened_s1})
                Qs[0, a] = reward + dis * np.max(Qs1)

            sess.run(train, feed_dict={X: flattened_s, Y: Qs})
            s = s1

        rList.append(step_count)
        print("Episode:{} step: {}".format(i, step_count))

        # If Average Score of last 10 is above 500
        if len(rList) > 10 and np.mean(rList[-10:]) > 500:
            break

        observation = env.reset()
        reward_sum = 0

        while True:
            env.render()

            flattened_s = np.reshape(observation, [1, input_size])
            Qs = sess.run(Qpred, feed_dict={X: flattened_s})
            a = np.argmax(Qs)

            observation, reward, done, _ = env.step(a)
            reward_sum += reward

            if done:
                print("Total score: {}".format(reward_sum))
                break
