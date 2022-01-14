from mujoco.ant_nrg import AntEnergyEnv

env = AntEnergyEnv()

# env is created, now we can use it: 
for episode in range(10): 
    obs = env.reset()
    for step in range(50):
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        nobs, reward, done, info = env.step(action)
        print(reward)
        #env.render()