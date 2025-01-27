import pybullet as p
import time
import pybullet_data
from math import sin, cos

physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally
p.setGravity(0, 0, -9.81)
groundId = p.loadURDF("plane.urdf")
robotStartPos = [0, 0, 1]
robotStartOrientation = p.getQuaternionFromEuler([1.5708, 0, -1.5708])
robotId = p.loadURDF("myrobot.urdf.txt", robotStartPos, robotStartOrientation)
mode = p.POSITION_CONTROL
jointIndex = 0 # first joint is number 0
for i in range(10000):
    p.setJointMotorControl2(robotId, jointIndex, controlMode=mode, targetPosition=0.25-0.25*cos(i*0.01))
    p.setJointMotorControl2(robotId, 1, controlMode=mode, targetPosition=0.25+0.25*cos(i*0.01))
    p.setJointMotorControl2(robotId, 2, controlMode=mode, targetPosition=0.25+0.25*cos(i*0.01))
    p.setJointMotorControl2(robotId, 3, controlMode=mode, targetPosition=0.25-0.25*cos(i*0.01))
    p.setJointMotorControl2(robotId, 4, controlMode=mode, targetPosition=0.25+0.25*cos(i*0.01))
    p.setJointMotorControl2(robotId, 5, controlMode=mode, targetPosition=0.25-0.25*cos(i*0.01))
    p.setJointMotorControl2(robotId, 6, controlMode=mode, targetPosition=0.25-0.25*cos(i*0.01))
    p.setJointMotorControl2(robotId, 7, controlMode=mode, targetPosition=0.25+0.25*cos(i*0.01))
    p.stepSimulation()
    time.sleep(1./240.)
robotPos, robotOrn = p.getBasePositionAndOrientation(robotId)
print(robotPos, robotOrn)
p.disconnect()