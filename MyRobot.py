import gym
import pybullet as p
import pybullet_data
import numpy as np
import time 

class MyRobotEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-100, high=100, shape=(21,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        self.physicsClient = p.connect(p.GUI)  # p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.groundId = p.loadURDF("plane.urdf")
        robotStartPos = [0, 0, 0.5]
        robotStartOrientation = p.getQuaternionFromEuler([1.5708, 0, -1.5708])
        self.robotId = p.loadURDF("myrobot.urdf.txt", robotStartPos, robotStartOrientation)
        self.start_time = time.time()
        self.previousPosition = 0
        
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.groundId = p.loadURDF("plane.urdf")
        robotStartPos = [0, 0, 0.5]
        robotStartOrientation = p.getQuaternionFromEuler([1.5708, 0, -1.5708])
        self.robotId = p.loadURDF("myrobot.urdf.txt", robotStartPos, robotStartOrientation)
        self.start_time = time.time()
        self.previousPosition = 0
        return self.getObservation()
    
    def step(self, action):
        for joint in range(p.getNumJoints(self.robotId)):
            p.setJointMotorControl2(self.robotId, joint, p.POSITION_CONTROL, targetPosition=action[8])
        p.stepSimulation()
        time.sleep(1./ 240.)  
        observation = self.getObservation()
        reward = self.calculateReward(observation)
        fall = self.fall(observation)
        info = self.getInfo(observation)
        return observation, reward, fall, info 
        
    def getObservation(self):
        position, orientation = p.getBasePositionAndOrientation(self.robotId)
        jointStates = p.getJointStates(self.robotId, range(p.getNumJoints(self.robotId)))
        jointPositions = [state[0] for state in jointStates]
        velocity, angularVelocity = p.getBaseVelocity(self.robotId)
        observation = np.concatenate([position, orientation, jointPositions, velocity, angularVelocity])
        return observation 
    
    def calculateReward(self, observation):
        position = observation[:3]
        distance = -100 if position[1] - self.previousPosition < 0.2 else position[1] - self.previousPosition
        self.previousPosition = position[1]
        deviations = -abs(position[0])
        velocity = observation[10:13]
        speed = 10 * velocity[1]
        roll, pitch, yaw = p.getEulerFromQuaternion(observation[3:7])
        stability = -1000 if roll < 0.785 or roll > 2.355 or abs(pitch) > 0.785 or position[2] < 0.07 else 0
        jointStates = p.getJointStates(self.robotId, range(p.getNumJoints(self.robotId)))
        energy = -0.1 * sum([abs(state[3]) for state in jointStates])
        reward = distance + deviations + speed + stability + energy
        return reward
    
    def fall(self, observation):
        position = observation[:3]
        current_time = time.time()
        if position[2] < 0.07: 
            return True
        if current_time - self.start_time > 60:
            return True
        return False
    
    def getInfo(self, observation):
        position = observation[:3]
        orientation = observation[3:7]
        jointStates = p.getJointStates(self.robotId, range(p.getNumJoints(self.robotId)))
        jointPositions = [state[0] for state in jointStates]
        velocity = observation[10:13]
        energy = sum([abs(state[3]) for state in jointStates]) 
        info = {
            'position': position,
            'orientation': orientation,
            'joint_positions': jointPositions,
            'velocity': velocity,
            'energy_consumption': energy 
        }
        return info