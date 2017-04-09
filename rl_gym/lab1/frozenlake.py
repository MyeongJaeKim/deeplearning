import rl_gym
from rl_gym.envs.registration import register
from colorama import init
from kbhit import KBHit

'''
https://www.youtube.com/watch?v=xvDAURQVDhk&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG&index=3
'''

init(autoreset=True)

register(
    id='FrozenLake-v3',
    entry_point='rl_gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery' : False}
)

env = rl_gym.make('FrozenLake-v3')
env.render()

while True:
    key = KBHit()
    action = key.getarrow()

    if action not in [0, 1, 2, 3]:
        print("Game Aborted!")
        break

    state, reward, done, info = env.step(action)
    env.render()

    print ("State: ", state, " Action: ", action, " Reward: ", reward, " Info: ", info)

    if done:
        print ("Finished with reward", reward)
        break
