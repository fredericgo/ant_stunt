from goal.ant_goal_special import AntGoalSpEnv

env = AntGoalSpEnv()

# env is created, now we can use it: 
for episode in range(10):
    goal = env.sample_goal()
    obs = env.reset()
    for step in range(50):
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        nobs, reward, done, info = env.step(action)
        env.render()