import MyRobot
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.reward = []
        self.timesteps = []
        self.mean = -np.inf
        self.path = "best.zip"
        
    def _on_step(self) -> bool: 
        if "episode" in self.locals["infos"][0]: 
            episode = self.locals["infos"][0]["episode"]["r"]
            self.reward.append(episode)
            self.timesteps.append(self.num_timesteps)
            
        mean = np.mean(self.reward[-100:]) if len(self.reward) >= 100 else np.mean(self.reward)
        if mean > self.mean: 
            self.mean = mean
            self.model.save(self.path)
        
        return True
    
    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.timesteps, self.reward, label="Reward")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("Proximal Policy Optimization")
        plt.legend()
        plt.savefig("ppo.png")
        plt.show()

env = MyRobot.MyRobotEnv()
callback = RewardCallback()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000, callback=callback)

model.save("last")

callback.plot()