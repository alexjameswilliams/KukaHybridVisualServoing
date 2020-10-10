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
        self.terminated = 0
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

        self.terminated = 0
        p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        p.setGravity(0, 0, -10)

        # Load KUKA iiwa Model
        start_pos = [0, 0, 0.001]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self._kuka = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation)

        # Set KUKA Joint angles
        jointPositions = [0.0, 0.0, 0.000000, 0.0, 0.000000, 0.0, 0.0]
        for jointIndex in range(p.getNumJoints(self._kuka)):
            p.resetJointState(self._kuka, jointIndex, jointPositions[jointIndex])
        self.setKukaJointAngles(jointPositions)

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
        numJoints = p.getNumJoints(self._kuka)

        lowerLimits = []
        upperLimits = []
        for joint in range(numJoints):
            info = p.getJointInfo(self._kuka, joint)
            lowerLimits.append(info[8])
            upperLimits.append(info[9])

        return lowerLimits, upperLimits


    # Set the joint angles of the kuka robot
    # jointPositions should be in radians
    def setKukaJointAngles(self, jointPositions):
        if len(jointPositions) != p.getNumJoints(self._kuka)
            return false
        else:
            p.setJointMotorControlArray(self._kuka, range(p.getNumJoints(self._kuka)), p.POSITION_CONTROL,
                                        jointPositions)
            return true


    # Get a camera image from end of Kuka robot
    # Credit: https://github.com/bulletphysics/bullet3/issues/1616
    def kuka_camera(self):

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


    def getObservation(self):
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
        self._observation = np_img_arr
        return self._observation

    def step(self, action):
        jointPositions = action
        self.setKukaJointAngles(jointPositions)
        self.kuka_camera()
        p.stepSimulation()

    def _termination(self):
        return

    def render(self, mode='human'):
        return

    def _termination(self):
        return

    def _reward(selfself):
        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
