# -*- coding: utf-8 -*-
"""
Created on Wed May 15 08:45 2024

@author: Daryl Choo Chia Ler
"""

import gym
import pybullet as p
import pybullet_data
import numpy as np
import time
from stable_baselines3 import PPO

class PyBulletEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(21,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        self.physicsClient = p.connect(p.GUI)  # Use p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.groundId = p.loadURDF("plane.urdf")
        robotStartPos = [0, 0, 1]
        robotStartOrientation = p.getQuaternionFromEuler([1.5708, 0, 1.5708])
        self.robotId = p.loadURDF("myrobot.urdf.txt", robotStartPos, robotStartOrientation)
        self.start_time = time.time()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.groundId = p.loadURDF("plane.urdf")
        robotStartPos = [0, 0, 1]
        robotStartOrientation = p.getQuaternionFromEuler([1.5708, 0, 1.5708])
        self.robotId = p.loadURDF("myrobot.urdf.txt", robotStartPos, robotStartOrientation)
        self.start_time = time.time()
        return self._get_observation()

    def step(self, action):
        # Apply action to the robot
        for joint in range(p.getNumJoints(self.robotId)):
            p.setJointMotorControl2(self.robotId, joint, p.POSITION_CONTROL, targetPosition=action[joint % 8])

        p.stepSimulation()
        time.sleep(1. / 240.)  # Control simulation speed

        obs = self._get_observation()
        reward = self._calculate_reward(obs)
        done = self._is_done(obs)
        info = self._get_info(obs)

        return obs, reward, done, info

    def _get_observation(self):
        position, orientation = p.getBasePositionAndOrientation(self.robotId)
        joint_states = p.getJointStates(self.robotId, range(p.getNumJoints(self.robotId)))
        joint_positions = [state[0] for state in joint_states]
        velocity, angular_velocity = p.getBaseVelocity(self.robotId)
        observation = np.concatenate([position, orientation, joint_positions, velocity, angular_velocity])
        return observation

    def _calculate_reward(self, obs):
        position = obs[:3]
        orientation = obs[3:7]
        initial_orientation = [0, 0, 0, 1]  # Quaternion for initial orientation
        distance_from_start = np.linalg.norm(position)
        orientation_reward = -np.linalg.norm(np.array(orientation) - np.array(initial_orientation))
        stability_penalty = -10 if position[2] < 0.5 else 0
        reward = distance_from_start + orientation_reward + stability_penalty
        return reward

    def _is_done(self, obs):
        position = obs[:3]
        current_time = time.time()
        if position[2] < 0.5:  # Falling condition
            return True
        if current_time - self.start_time > 60:  # Time limit condition
            return True
        return False

    def _get_info(self, obs):
        position = obs[:3]
        orientation = obs[3:7]
        joint_states = p.getJointStates(self.robotId, range(p.getNumJoints(self.robotId)))
        joint_positions = [state[0] for state in joint_states]
        velocity = obs[10:13]
        energy_consumption = sum([abs(state[3]) for state in joint_states])  # Summing absolute torques as a proxy for energy consumption
        info = {
            'position': position,
            'orientation': orientation,
            'joint_positions': joint_positions,
            'velocity': velocity,
            'energy_consumption': energy_consumption
        }
        return info

# Train the agent using PPO
env = PyBulletEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_pybullet_model")