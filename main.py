import MyRobot
from stable_baselines3 import PPO

env = MyRobot.MyRobotEnv()
model = PPO.load("best.zip", env=env)
observation = env.reset()

for i in range(10000): 
    action, states = model.predict(observation, deterministic=True)
    observation, reward, fall, info = env.step(action)
    
    if fall: 
        observation = env.reset()