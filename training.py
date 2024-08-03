# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:41:15 2024

@author: Acer
"""

import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
from stable_baselines3 import PPO

class PyBulletEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Connect to the physics server
        self.physicsClient = p.connect(p.GUI)  # Use p.GUI for graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.groundId = p.loadURDF("plane.urdf")
        robotStartPos = [0, 0, 1]
        robotStartOrientation = p.getQuaternionFromEuler([1.5708, 0, 1.5708])
        self.robotId = p.loadURDF("myrobot.urdf.txt", robotStartPos, robotStartOrientation)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.groundId = p.loadURDF("plane.urdf")
        robotStartPos = [0, 0, 1]
        robotStartOrientation = p.getQuaternionFromEuler([1.5708, 0, 1.5708])
        self.robotId = p.loadURDF("myrobot.urdf.txt", robotStartPos, robotStartOrientation)
        return self._get_observation()

    def step(self, action):
        # Apply action to the robot
        # Here you should implement the code to control the robot using the action provided

        p.stepSimulation()
        time.sleep(1. / 240.)  # Control simulation speed

        obs = self._get_observation()
        reward = 0  # You should define how to calculate the reward
        done = False  # You should define the termination condition
        info = {}  # Additional information, if needed

        # Render the environment
        p.removeAllUserDebugItems()
        # Example visualization, adjust as needed
        #p.addUserDebugLine([0, 0, 0], [1, 1, 1], [1, 0, 0], 3)

        return obs, reward, done, info

    def _get_observation(self):
        # Here you should implement the code to retrieve the observation from the environment
        observation = np.zeros(10)  # Placeholder observation
        return observation

# Create an instance of your PyBullet environment
env = PyBulletEnv()

# Load the trained policy from file
model = PPO.load("ppo_pybullet_model.zip", env=env)

# Reset the environment
obs = env.reset()

# Simulate for a certain number of steps
for _ in range(1000):  
    # Use the trained policy to choose an action
    action, _ = model.predict(obs, deterministic=True)
    
    # Take a step in the environment
    obs, reward, done, _ = env.step(action)
    
    # Check if the episode is done
    if done:
        obs = env.reset()

# Close the environment
env.close()