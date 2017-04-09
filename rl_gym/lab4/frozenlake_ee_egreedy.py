import rl_gym
import numpy as np
import matplotlib.pyplot as plt
from rl_gym.envs.registration import register

'''
https://www.youtube.com/watch?v=VYOq-He90bE&index=7&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG
'''

register(
    id='FrozenLake-v3',
    entry_point='rl_gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4',
            'is_slippery' : False}
)

env = rl_gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])
dis = .99
num_episodes = 2000

rList = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action  = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

        new_state, reward, done, _ = env.step(action)

        Q[state, action] = reward + dis * np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success Rate:" + str(sum(rList) / num_episodes))
print("Final Q-Table")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rList)), rList, color='b', alpha=0.4)
plt.show()