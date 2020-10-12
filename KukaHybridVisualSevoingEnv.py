import os
import numpy as np
import time

import pybullet as p
import pybullet_data

import gym
from gym.utils import seeding
from gym import spaces

maxSteps = 1000

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class KukaHybridVisualSevoingEnv(gym.Env):

    # Initialise Environment
    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False):
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._width = 341
        self._height = 256
        self._isDiscrete = isDiscrete
        self.terminated = False
        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        # timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")

        self.seed()
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())  #todo do I need this and the rooturdf line?
        self.reset()
        observationDim = len(self.getObservation())

        observation_high = np.array([np.finfo(np.float32).max] * observationDim)
        if (self._isDiscrete):
            self.action_space = spaces.Discrete(7)
        else:
            action_dim = 3
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self._height, self._width, 4),
                                            dtype=np.uint8)
        self.viewer = None

    # Disconnect from physics server and end simulation
    def __del__(self):
        p.disconnect()

    # Initialise Environment
    def reset(self):

        self.terminated = False
        p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        p.setGravity(0, 0, -10)



        # Load objects
        self.generateRobot()
        self.floor_id = self.generateFloor()
        self.target_id = self.generateTarget()

        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getObservation()
        return np.array(self._observation)

    @property
    # Returns the Kuka's current joint state including Position, Velocity,
    # reaction forces and applied motor torque
    def getKukaJointStates(self):
        numJoints = p.getNumJoints(self._kuka)

        return p.getJointStates(range(numJoints))

    @property
    # Returns the lower and upper joint limits for each joint on the iiwa in radians as two arrays
    def getKukaJointLimits(self):

        lowerLimits = []
        upperLimits = []
        for joint in range(p.getNumJoints(self._kuka)):
            info = p.getJointInfo(self._kuka, joint)
            lowerLimits.append(info[8])
            upperLimits.append(info[9])

        return lowerLimits, upperLimits

    # Set the joint angles of the kuka robot
    # jointPositions should be in radians
    def setKukaJointAngles(self, jointPositions):

        # Check number of joint inputs is correct
        if len(jointPositions) != p.getNumJoints(self._kuka):
            return False
        else:
            # Check inputs are within maximum and minimum joint limits and amend if necessary
            lowerLimits, upperLimits = self.getKukaJointLimits
            for joint in range(p.getNumJoints(self._kuka)):
                if jointPositions[joint] < lowerLimits[joint]:
                    jointPositions[joint] = lowerLimits[joint]

                elif jointPositions[joint] > upperLimits[joint]:
                    jointPositions[joint] = upperLimits[joint]

            p.setJointMotorControlArray(self._kuka, range(p.getNumJoints(self._kuka)), p.POSITION_CONTROL,
                                        jointPositions)
            return True

    # Returns an RGB image from the Eye In Hand camera on the robot's end effector
    # Credit: https://github.com/bulletphysics/bullet3/issues/1616
    def _getEyeInHandCamera(self):

        fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

        # Center of mass position and orientation (of link-7)
        com_p, com_o, _, _, _, _ = p.getLinkState(self._kuka, 6, computeForwardKinematics=True)
        rot_matrix = p.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (0, 1, 0)  # y-axis
        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
        img = p.getCameraImage(1000, 1000, view_matrix, projection_matrix)
        return img

    # Returns an RGB image from the Eye To Hand camera
    def _getEyeToHandCamera(self):

        camEyePos = [0.03, 0.236, 0.54]
        distance = 1.06
        pitch = -56
        yaw = 258
        roll = 0
        upAxisIndex = 2
        camInfo = p.getDebugVisualizerCamera()
        viewMat = camInfo[2]
        viewMat = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, yaw, pitch, roll, upAxisIndex)
        """viewMat = [
			-0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722,
			-0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843,
			0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0
		]
		"""
        # projMatrix = camInfo[3]#[0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
        projMatrix = [
            0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
            -0.02000020071864128, 0.0
        ]

        img_arr = p.getCameraImage(width=self._width,
                                   height=self._height,
                                   viewMatrix=viewMat,
                                   projectionMatrix=projMatrix)
        rgb = img_arr[2]
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        return np_img_arr


    # Observation consists of four components which are collected and returned via this function.
    # image_eih: Camera image from robot's end effector (Eye In Hand)
    # image_eth: Camera image overlooking scene (Eye to Hand)
    # joint_positions: Robot joint positions
    # joint_velocities: Robot joint velocities
    def getObservation(self):

        observation = []
        positions = []
        velocities = []

        # Get camera observation data
        image_eih = self._getEyeInHandCamera()
        image_eth = self._getEyeToHandCamera()

        # Get robotic joints observation data
        jointRange = np.arange(0, p.getNumJoints(self._kuka))
        jointStates = p.getJointStates(self._kuka, jointRange)
        for joint in jointStates:
            positions.append(joint[0])
            velocities.append(joint[1])

        observation.append(image_eih)
        observation.append(image_eth)
        observation.append(positions)
        observation.append(velocities)

        return observation


    def step(self, action):
        jointPositions = action
        self.setKukaJointAngles(jointPositions)
        p.stepSimulation()

        observation = self.getObservation()
        reward = self._reward()
        done = self._termination()

        return np.array(self._observation), reward, done

    def render(self, mode='human'):
        return

    # Termination on collision, goal achievement, or run out of time
    def _termination(self):
        if (self.terminated):
            return True
        return False

    # Reward Function has 5 possible subfunctions which are incorporated depending on sparseness of reward signal
    # These subfunctions can be activated by setting the following flags
    # rGoal: Binary reward signal of whether reward achieved or not
    # rCollision: Binary reward signal of whether collision detected
    # rTime: Incremental penalty for each timestep within allowed timelimit
    # rTranslation: Incremental Reward for translation towards target position
    # rRotation: Incremental penalty for deviation from target orientation

    # By default, rGoal, rCollision, and rTime are enabled and rTranslation and rRotation are disabled
    def _reward(self, rGoal=True, rCollision=True, rTime=True, rRotation=False, rTranslation=False):

        # rGoal
        # rCollision

        # 1) Check if collision detected
        contact = p.getContactPoints(self._kuka,self.floor_id)
        if contact:
            # if detected then end simulation and return negative reward
            self.terminated = True
            reward = -1

        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Generate random coordinates for the episodic target and place as a flat urdf model in environment
    def generateTarget(self):

        # Generate 4 random numbers to choose 2 for the x and y coordinates of the target
        # This is to avoid the robots base and ensure targets are within robot's reach
        kuka_min_range = .420  # approximation based on robot documentation
        kuka_max_range = .800  # approximation based on robot documentation
        ranges = np.random.uniform(kuka_min_range, kuka_max_range, 4)
        x_ranges = np.stack((ranges[0], -ranges[1]))
        y_ranges = np.stack((ranges[2], -ranges[3]))
        target_x = np.random.choice(x_ranges)
        target_y = np.random.choice(y_ranges)
        self.target_position = [target_x,target_y]

        print('target pos = ' + str(np.multiply(self.target_position,1000)))

        # Load URDF file and colour
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName='target.obj',
            rgbaColor=[1,0,0,1], # red
            meshScale=[1, 1, 1])

        # Place object at generated coordinates
        target_id = p.createMultiBody(
            baseVisualShapeIndex=visualShapeId,
            basePosition=[target_x, target_y, 0.0001],
            baseOrientation=p.getQuaternionFromEuler([np.deg2rad(90), 0, 0]))

        return target_id

    # Load floor urdf file and place in environment
    def generateFloor(self):

        # Load URDF file and colour
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName='floor.obj',
            rgbaColor=[0.6, 0.6, 0.6, 1], # grey
            meshScale=[1, 1, 1])

        # Ensure floor can be collided with
        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName='floor.obj',
            meshScale=[1, 1, 1])

        # Merge collision and visual profile and place object
        floor = p.createMultiBody(
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=[0,0,0],
            baseOrientation=p.getQuaternionFromEuler([np.deg2rad(90), 0, 0]))

        return floor

    def generateRobot(self):
        # Load KUKA iiwa Model
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self._kuka = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation)

        # Set KUKA Joint angles
        jointPositions = [0.0, 0.0, 0.000000, 0.0, 0.000000, 0.0, 0.0]
        for jointIndex in range(p.getNumJoints(self._kuka)):
            p.resetJointState(self._kuka, jointIndex, jointPositions[jointIndex])
        self.setKukaJointAngles(jointPositions)


