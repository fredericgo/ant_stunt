from mujoco.ant_takeoff import AntTakeoff
import numpy as np

env = AntTakeoff(v0=np.array([3, 3, 10]))

# env is created, now we can use it: 
for episode in range(1): 
    obs = env.reset()
    for step in range(100):
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        nobs, reward, done, info = env.step(action)
        print(reward)
        env.render()