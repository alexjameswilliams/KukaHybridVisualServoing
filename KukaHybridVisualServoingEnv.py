import os
import numpy as np

import pybullet as p
import pybullet_data

from gym.utils import seeding
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# todo verify ranges in simulation and minimum height
KUKA_MIN_RANGE = .400   # approximation based on robot documentation
KUKA_MAX_RANGE = .780   # approximation based on robot documentation
KUKA_VELOCITY_LIMIT = 1.0    # Maximum velocity of robotic joints. If not lmiited then robot goes out of its joint limit


class KukaHybridVisualServoingEnv(py_environment.PyEnvironment):

    # Initialise Environment
    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision: bool = True,
                 renders: bool = False,
                 isDiscrete: bool = False,
                 eih_input: bool = True,
                 eth_input: bool = True,
                 position_input: bool = True,
                 velocity_input: bool = True,
                 eih_camera_resolution: int = 120,
                 eth_camera_resolution: int = 120,
                 image_depth: int = 3,
                 steps: int = 20,
                 discount = 1.0,
                 rGoal=True,
                 rCollision=True,
                 rTime=True,
                 rRotation=False,
                 rTranslation=False):
        self._timeStep = 1. / 240.
        self._simulationStepsPerTimeStep = 24
        self.max_steps = steps
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._isDiscrete = isDiscrete
        self._eih_input = eih_input
        self._eth_input = eth_input
        self._image_depth = image_depth
        self._position_input = position_input
        self._velocity_input = velocity_input
        self._eih_camera_resolution: int = eih_camera_resolution
        self._eth_camera_resolution: int = eth_camera_resolution
        self.discount = discount
        self.rGoal = rGoal
        self.rCollision = rCollision
        self.rTime = rTime
        self.rRotation = rRotation
        self.rTranslation = rTranslation

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
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())  # todo do I need this and the rooturdf line?
        self._reset()
        self.viewer = None
        self._action_spec = array_spec.BoundedArraySpec(shape=(1,7), dtype=np.float32, minimum=0.0, maximum=1.0, name='joint positions')
        self._observation_spec = self.observation_spec()


    # Disconnect from physics server and end simulation
    def __del__(self):
        p.disconnect()

    # Initialise Environment
    def _reset(self):

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
        return ts.restart(observation=self._observation)
        #return np.array(self._observation)

    @property
    # Returns the Kuka's current joint state including Position, Velocity,
    # reaction forces and applied motor torque
    def getKukaJointStates(self):
        numJoints = p.getNumJoints(self._kuka)

        return p.getJointStates(range(numJoints))

    @property
    # Returns the lower and upper joint limits for each joint on the iiwa in radians as two arrays
    def getKukaJointLimits(self):

        lower_limits = []
        upper_limits = []
        for joint in range(p.getNumJoints(self._kuka)):
            info = p.getJointInfo(self._kuka, joint)
            lower_limits.append(info[8])
            upper_limits.append(info[9])

        return lower_limits, upper_limits

    @property
    # Returns the minimum and maximum velocity limits for each joint on the iiwa
    def getKukaVelocityLimits(self):
        minimum_velocities = []
        maximum_velocities = []
        for joint in range(p.getNumJoints(self._kuka)):
            info = p.getJointInfo(self._kuka, joint)
            minimum_velocities.append(-info[11])
            maximum_velocities.append(info[11])

        return minimum_velocities, maximum_velocities


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

        eih_res = self._eih_camera_resolution[0]
        eih_dep = self._image_depth[0]

        # todo fine tune resolution
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
        img = p.getCameraImage(eih_res, eih_res, view_matrix,
                               projection_matrix)
        rgba_data = img[2]
        return self._pixelArray2RGBArray(rgba_data, eih_dep, eih_res)
        # todo add mono image return

    # Returns an RGB image from the Eye To Hand camera
    # todo possibly add manipulation of camera position and resolution
    def _getEyeToHandCamera(self):

        eth_res = self._eth_camera_resolution[0]
        eth_dep = self._image_depth[0]

        # todo fine tune positioning and resolution
        camEyePos = [0.03, 0.236, 0.54]
        distance = 1.06
        pitch = -56
        yaw = 258
        roll = 0
        upAxisIndex = 2
        camInfo = p.getDebugVisualizerCamera()
        #print("width,height")
        #print(camInfo[0])
        #print(camInfo[1])
        #print("viewMatrix")
        #print(camInfo[2])
        #print("projectionMatrix")
        #print(camInfo[3])
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

        img_arr = p.getCameraImage(eth_res, eth_res, viewMat, projMatrix)
        # rgb = img_arr[2]
        # np_img_arr = np.reshape(rgb, (*self._eth_camera_resolution, *self._eth_camera_resolutionn, 4))
        rgba_data = img_arr[2]
        return self._pixelArray2RGBArray(rgba_data, eth_dep, eth_res)
        # todo add mono image return

    # Converts Matrices in the format where they are stored as arrays of RGBA values for individual pixels, into separate R,G,B arrays while maintaining image dimensions
    def _pixelArray2RGBArray(self, rgba_seq, img_dep, img_res):
        r, g, b = [], [], []
        for row in rgba_seq:
            for pixel in row:
                r.append(pixel[0])
                g.append(pixel[1])
                b.append(pixel[2])

        rgb = [r, g, b]
        # Split into three RGB layers
        return np.array(rgb).reshape(img_dep, img_res, img_res)

    # Observation consists of four components which are collected and returned via this function.
    # image_eih: Camera image from robot's end effector (Eye In Hand)
    # image_eth: Camera image overlooking scene (Eye to Hand)
    # joint_positions: Robot joint positions
    # joint_velocities: Robot joint velocities
    def getObservation(self):

        observation= {}

        # Get camera observation data
        if self._eih_input:
            eih_dep = self._image_depth[0]
            eih_res = self._eih_camera_resolution[0]
            image_eih = self._getEyeInHandCamera()
            observation.update({'eih_image': np.array(image_eih, dtype=np.float32)})
            # print("OBSERVATION EIH:")
            # print(image_eih)

        if self._eth_input:
            eth_dep = self._image_depth[0]
            eth_res = self._eth_camera_resolution[0]
            image_eth = self._getEyeToHandCamera()
            observation.update({'eth_image': np.array(image_eth, dtype=np.float32)})

            # print("OBSERVATION ETH:")
            # print(image_eth)

        # Get robotic joints observation data
        if self._position_input:
            positions = []
            jointRange = np.arange(0, p.getNumJoints(self._kuka))
            jointStates = p.getJointStates(self._kuka, jointRange)
            for joint in jointStates:
                positions.append(joint[0])

            observation.update({'joint_positions': np.array(positions, dtype=np.float32)})

            # print("OBSERVATION POS:")
            # print(positions)
        if self._velocity_input:
            velocities = []
            jointRange = np.arange(0, p.getNumJoints(self._kuka))
            jointStates = p.getJointStates(self._kuka, jointRange)
            for joint in jointStates:
                velocities.append(joint[1])

            observation.update({'joint_velocities': np.array(velocities, dtype=np.float32)})
            # print("OBSERVATION VEL:")
            # print(velocities)

        print("OBSERVATION:")
        print(observation)

        return observation

    # Converts a normalised joint angle value between [0..1] into joint range equivalent radian value
    def normalised_action_to_joint_angles(self, action):

        actual_joint_values = []
        # Check number of joint inputs is correct
        if len(action) != p.getNumJoints(self._kuka):
            return False

        # Check inputs are within maximum and minimum joint limits and amend if necessary
        lowerLimits, upperLimits = self.getKukaJointLimits
        for joint in np.arange(p.getNumJoints(self._kuka)):
            range = upperLimits[joint] + lowerLimits[joint]
            value = lowerLimits[joint] + (range * action[joint])
            actual_joint_values.append(value)

        return actual_joint_values

    # Normalises the joint angle to a value between [0..1]
    def normalise_joint_angles(self, joint_positions):

        normalised_values = []
        # Check number of joint inputs is correct
        if len(joint_positions) != p.getNumJoints(self._kuka):
            return False

        # Check inputs are within maximum and minimum joint limits and amend if necessary
        lowerLimits, upperLimits = self.getKukaJointLimits
        for joint in np.arange(p.getNumJoints(self._kuka)):
            normal_value = (joint_positions[joint] / np.abs(upperLimits[joint])) * 0.5
            normalised_values.append(0.5 + normal_value)

        return normalised_values

    # Advance simulation and collect observation, reward, and termination data
    def _step(self, action):

        jointPositionTarget = self.normalised_action_to_joint_angles(action)
        print('JPT')
        print(jointPositionTarget)
        self.setKukaJointAngles(jointPositionTarget)
        for step in np.arange(0, self._simulationStepsPerTimeStep):
            p.stepSimulation()

        self._envStepCounter += 1
        done = self._termination()
        observation = self.getObservation()
        reward = self._reward(self.rGoal, self.rCollision, self.rTime, self.rRotation, self.rTranslation)

        if done:
            return ts.termination(observation=observation, reward=reward)
        else:
            return ts.transition(observation=observation, reward=reward, discount=self.discount)

    def render(self, mode='human'):
        return

    # Checks if a condition for termination has been met
    # Termination can be triggered by collision, goal achievement, or when the maximum number of steps have been reached
    def _termination(self):
        if (self.terminated or self._envStepCounter > self.max_steps):
            self.terminated = True
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

        max_goal_magnitude = 1.0
        reward = 0.0
        if rGoal:
            self.translational_tolerance = 0.001  # 1mm
            self.angular_tolerance = 1.0  # 1 degree
            self.vertical_distance = 0.05  # 5cm

            # 1) Calculate Kuka End Effector position and orientation (link 7)
            effector_position, effector_orientation, _, _, _, _ = p.getLinkState(self._kuka, 6,
                                                                                 computeForwardKinematics=True)

            # 2) Retrieve Target Position
            target_position, target_orientation = p.getBasePositionAndOrientation(self.target_id)
            target_position = np.asarray(target_position)
            target_position[2] = target_position[2] + self.vertical_distance

            # 3) Calculate Euclidean distance between end effector and target
            distance = np.linalg.norm(target_position - np.asarray(effector_position))
            # print('Distance = ' + str(distance))

            # 4) Calculate orientation matrix between effector orientation and target orientation
            target_orientation = np.array(p.getEulerFromQuaternion(target_orientation))
            target_orientation[0] += np.deg2rad(90)
            target_orientation = p.getQuaternionFromEuler(target_orientation)
            orientation_diff = p.getDifferenceQuaternion(target_orientation, effector_orientation)

            # 5) Calculate cumulative rotational error
            orientation_diff = p.getEulerFromQuaternion(orientation_diff)
            cumulative_error = 0
            for orientation in orientation_diff:
                cumulative_error += np.abs(orientation)

            # 6) Check if within dimensional tolerances, if true then goal achieved
            if (np.abs(distance) <= self.translational_tolerance) and (cumulative_error <= self.angular_tolerance):
                print('Goal Achieved')
                reward += max_goal_magnitude
                self.terminated = True

            # ? 7) Verify that target is in camera?

        if rTime and self.terminated:
            # 1) Reduce maximum reward by an amount proportional to the number of timesteps
            reward += -((max_goal_magnitude / self.max_steps) * self._envStepCounter)

        if rCollision:
            # 1) Check if collision detected
            contact = p.getContactPoints(self._kuka, self.floor_id)
            if contact:
                # 2) if detected then end simulation and return maximum penalty
                print('Collision Detected')
                reward = -max_goal_magnitude
                self.terminated = True

        if rTranslation and not self.terminated:
            # 1) Calculate Kuka End Effector position and orientation (link 7)
            # todo possibly link in to rGoal to avoid calling twice
            effector_position, _, _, _, _, _ = p.getLinkState(self._kuka, 6, computeForwardKinematics=True)

            # 2) Retrieve Target Position
            target_position, _ = p.getBasePositionAndOrientation(self.target_id)
            target_position = np.asarray(target_position)
            target_position[2] = target_position[2] + self.vertical_distance

            # 3) Calculate Euclidean distance between end effector and target
            distance = np.linalg.norm(target_position - np.asarray(effector_position))

            # 4) Return up to 0.4 of maximum reward until within tolerance range
            max_possible_distance = (
                                                2 * KUKA_MAX_RANGE) - self.translational_tolerance  # todo this may be a bit of a blunt implementation - better heuristic?
            reward = 0.4 * max_goal_magnitude * (
                        (max_possible_distance - np.abs(distance) + self.tolerance) / max_possible_distance)
            if reward > 0.4 * max_goal_magnitude:
                reward += 0.4 * max_goal_magnitude

        if rRotation and not self.terminated:
            # 1) Calculate orientation matrix between effector orientation and target orientation
            target_orientation = np.array(p.getEulerFromQuaternion(target_orientation))
            target_orientation[0] += np.deg2rad(90)
            target_orientation = p.getQuaternionFromEuler(target_orientation)
            orientation_diff = p.getDifferenceQuaternion(target_orientation, effector_orientation)

            # 2) Calculate cumulative rotational error
            orientation_diff = p.getEulerFromQuaternion(orientation_diff)
            cumulative_error = 0
            for orientation in orientation_diff:
                cumulative_error += np.abs(orientation)

            # 4) Return up to 0.4 of maximum reward until within tolerance range
            max_possible_rotation = (
                                                2 * 180) - self.angular_tolerance  # todo this may be a bit of a blunt implementation - better heuristic?
            reward = 0.4 * max_goal_magnitude * ((max_possible_rotation - np.abs(
                cumulative_error) + self.angular_tolerance) / max_possible_rotation)
            if reward > 0.4 * max_goal_magnitude:
                reward += 0.4 * max_goal_magnitude

        return reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Generate random coordinates for the episodic target and place as a flat urdf model in environment
    def generateTarget(self):

        # Generate 4 random numbers to choose 2 for the x and y coordinates of the target
        # This is to avoid the robots base and ensure targets are within robot's reach
        ranges = np.random.uniform(KUKA_MIN_RANGE, KUKA_MAX_RANGE, 4)
        x_ranges = np.stack((ranges[0], -ranges[1]))
        y_ranges = np.stack((ranges[2], -ranges[3]))
        target_x = np.random.choice(x_ranges)
        target_y = np.random.choice(y_ranges)
        self.target_position = [target_x, target_y]

        print('target pos = ' + str(np.multiply(self.target_position, 1000)))

        # Load URDF file and colour
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName='target.obj',
            rgbaColor=[1, 0, 0, 1],  # red
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
            rgbaColor=[0.6, 0.6, 0.6, 1],  # grey
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
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([np.deg2rad(90), 0, 0]))

        return floor

    def generateRobot(self):
        # Load KUKA iiwa Model
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self._kuka = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation)

        # Set KUKA Joint angles
        jointPositions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for jointIndex in np.arange(p.getNumJoints(self._kuka)):
            p.resetJointState(self._kuka, jointIndex, jointPositions[jointIndex])
            p.setJointMotorControl2(bodyUniqueId=self._kuka, jointIndex=jointIndex, controlMode=p.POSITION_CONTROL, maxVelocity=KUKA_VELOCITY_LIMIT)
        self.setKukaJointAngles(jointPositions)


    def observation_spec(self):
        """Return observation_spec.
        Observation spec consists of two RGB images of (camera_resolution x camera_resolution) x 3 pixel values in the range 0..255
         and two 7 dimensional vectors"""

        lower_limits, upper_limits = self.getKukaJointLimits
        minimum_velocities, maximum_velocities = self.getKukaVelocityLimits

        eih_res = self._eih_camera_resolution
        eih_dep = self._image_depth
        eth_res = self._eth_camera_resolution
        eth_dep = self._image_depth

        observation_spec: array_spec.ArraySpec = {}

        if self._eih_input:
            observation_spec.update({'eih_image': array_spec.BoundedArraySpec(shape=(*eih_dep, *eih_res, *eih_res), dtype=np.float32, minimum=0,maximum=255)})

        
        if self._eth_input:
            observation_spec.update({'eth_image': array_spec.BoundedArraySpec(shape=(*eth_dep, *eth_res, *eth_res), dtype=np.float32, minimum=0,maximum=255)})
            

        if self._position_input:
            observation_spec.update({'joint_positions': array_spec.BoundedArraySpec(shape=(7,), dtype=np.float32,
                                                                                    minimum=lower_limits,
                                                                                    maximum=upper_limits)})
        if self._velocity_input:
            observation_spec.update({'joint_velocities': array_spec.BoundedArraySpec(shape=(7,), dtype=np.float32)})
                                                                                     #minimum=minimum_velocities,
                                                                                  #maximum=maximum_velocities)})
        return observation_spec

    def action_spec(self):
        """Return action_spec.
        Action spec consists of normalised values for the desired joint positions for the next time step."""

        return array_spec.BoundedArraySpec(shape=(7,), dtype=np.float32, minimum=0.0, maximum=1.0, name='joint positions')
