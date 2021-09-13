import os
import numpy as np

import pybullet as p
import pybullet_data

from gym.utils import seeding
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# todo verify ranges in simulation and minimum height
KUKA_MIN_RANGE = .400               # approximation based on robot documentation
KUKA_MAX_RANGE = .780               # approximation based on robot documentation
KUKA_VELOCITY_LIMIT = 1.0           # Maximum velocity of robotic joints. If not lmiited then robot goes out of its joint limit
MAX_REWARD_MAGNITUDE = 1.0          # Maximum magnitude of a positive or negative reward i.e. normalised to [-1,1]
TIME_PENALTY = 0.05                 # Penalty applied per timestep if time reward function activated. todo decide if this should be a fraction of the number of timesteps instead
POSITION_REWARD_CONSTANT = 0.4      # Maximum fraction of optimal reward that can be awarded for positional alignment if positional reward activated. Must be less than 0.5.
ROTATION_REWARD_CONSTANT = 0.4      # Maximum fraction of optimal reward that can be awarded for rotational alignment if rotational reward activated. Must be less than 0.5.

SIMULATION_STEP_DELTA = 1. / 240.   # Time between each simulation step
SIMULATION_STEPS_PER_TIMESTEP = 24  # Number of simulation steps in one algorithmic timestep

GOAL_SIZE = 5                       # Size of goal in visualiser

class KukaHybridVisualServoingEnv(py_environment.PyEnvironment):

    # Initialise Environment
    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision: bool = True,
                 renders: bool = False,
                 isDiscrete: bool = False, #todo I could add functionality for this in
                 eih_input: bool = True,
                 seed: int = None,
                 eth_input: bool = True,
                 position_input: bool = True,
                 velocity_input: bool = True,
                 eih_camera_resolution: int = 120,
                 eth_camera_resolution: int = 120,
                 eth_channels: int = 3,
                 eih_channels: int = 3,
                 timesteps: int = 50,
                 discount: float = 1.0,
                 reward_goal=True,
                 reward_collision=True,
                 reward_time=True,
                 reward_rotation=True,
                 reward_position=True,
                 normalise_observation=True): #todo add target behaviour parameters (shape, resolution, random etc.)
        self.max_steps = timesteps
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._timestep_count = 0
        self._renders = renders
        self._isDiscrete = isDiscrete
        self._eih_input = eih_input
        self._eth_input = eth_input
        self._eth_channels = eth_channels
        self._eih_channels = eih_channels
        self.eih_rgba = []
        self.eth_rgba = []
        self._position_input = position_input
        self._velocity_input = velocity_input
        self._eih_camera_resolution: int = eih_camera_resolution
        self._eth_camera_resolution: int = eth_camera_resolution
        self.discount = discount
        self.reward_goal = reward_goal
        self.reward_collision = reward_collision
        self.reward_time = reward_time
        self.reward_position = reward_position
        self.reward_rotation = reward_rotation
        self.normalise_observation = normalise_observation

        #todo could set these in the constructor for curriculum learning experiments
        self.positional_tolerance = 0.001  # 1mm
        self.rotational_tolerance = 1.0  # 1 degree
        self.vertical_distance = 0.05  # 5cm

        self.goal_achieved = False
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

        self.seed(seed)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())  # todo do I need this and the rooturdf line?
        self._reset()
        self.viewer = None
        self._action_spec = self.action_spec()
        self._observation_spec = self.observation_spec()


    # Disconnect from physics server and end simulation
    def __del__(self):
        p.disconnect()

    # Initialise Environment
    def _reset(self):

        self.goal_achieved = False
        self.terminated = False
        p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(SIMULATION_STEP_DELTA)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
        p.setGravity(0, 0, -10)


        # Load objects
        self.generateRobot()
        self.floor_id = self.generateFloor()
        self.target_id = self.generateGoal(size=GOAL_SIZE)

        self._timestep_count = 0
        p.stepSimulation()
        self._observation = self.getObservation()
        return ts.restart(observation=self._observation)

    def close(self):
        if self._pybullet_client:
            self._pybullet_client.disconnect()
            self._pybullet_client = None
            

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

        eih_res = self._eih_camera_resolution
        eih_dep = self._eih_channels

        # Set up camera positioning
        fov, aspect, nearplane, farplane = 50, 1.0, 0.01, 100
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

        self.eih_rgba = img[2]
        return self._pixelArray2RGBArray(self.eih_rgba, eih_dep, eih_res)
        # todo add mono image return

    # Returns an RGB image from the Eye To Hand camera
    # todo possibly add manipulation of camera position and resolution in method call
    def _getEyeToHandCamera(self):

        eth_res = self._eth_camera_resolution
        eth_dep = self._eth_channels

        # Set up camera positioning
        camEyePos = [0, 0, 0.1]
        distance = 1.3
        pitch = -65
        yaw = 275
        roll = 90
        upAxisIndex = 2
        camInfo = p.getDebugVisualizerCamera()
        viewMat = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, yaw, pitch, roll, upAxisIndex)
        projMatrix = [
            0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
            -0.02000020071864128, 0.0
        ]

        img_arr = p.getCameraImage(eth_res, eth_res, viewMat, projMatrix)
        # todo add mono image return
        self.eth_rgba = img_arr[2]
        return self._pixelArray2RGBArray(self.eth_rgba, eth_dep, eth_res)

    # Returns all current joint positions and velocities
    def _getJointStates(self):
        positions = []
        velocities = []
        jointRange = np.arange(0, p.getNumJoints(self._kuka))
        jointStates = p.getJointStates(self._kuka, jointRange)
        for joint in jointStates:
            positions.append(joint[0])
            velocities.append(joint[1])

        return positions, velocities


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

    # Observation consists of up to four components which are collected and returned via this function.
    # Observations will be normalised to the range [0..1] if the property self.normalise_observation is set to True.
    # Observation Components are:
    # image_eih: Camera image from robot's end effector (Eye In Hand)
    # image_eth: Camera image overlooking scene (Eye to Hand)
    # joint_positions: Robot joint positions
    # joint_velocities: Robot joint velocities
    def getObservation(self):

        observation = {}

        # Get camera observation data
        if self._eih_input:
            image_eih = self._getEyeInHandCamera()
            if self.normalise_observation:
                image_eih = self.normaliseImageValues(image_eih)
            observation.update({'eih_image': np.array(image_eih, dtype=np.float32)})

        if self._eth_input:
            image_eth = self._getEyeToHandCamera()
            if self.normalise_observation:
                image_eth = self.normaliseImageValues(image_eth)
            observation.update({'eth_image': np.array(image_eth, dtype=np.float32)})

        # Get robotic joints observation data
        if self._position_input or self._velocity_input:
            joint_positions, joint_velocities = self._getJointStates()
            if self._position_input:
                if self.normalise_observation:
                    joint_positions = self.normaliseJointAngles(joint_positions)
                observation.update({'joint_positions': np.array(joint_positions, dtype=np.float32)})

            if self._velocity_input:
                if self.normalise_observation:
                    joint_velocities = self.normaliseJointVelocities(joint_velocities)
                observation.update({'joint_velocities': np.array(joint_velocities, dtype=np.float32)})

        return observation

    # Converts a normalised joint angle value between [0..1] into the joint range-specific equivalent radian value
    def normalisedAction2JointAngles(self, action):

        actual_joint_values = []
        # Check number of joint inputs is correct
        if len(action) != p.getNumJoints(self._kuka):
            return False

        # Check inputs are within maximum and minimum joint limits and amend if necessary
        lowerLimits, upperLimits = self.getKukaJointLimits
        for joint in np.arange(p.getNumJoints(self._kuka)):
            range = upperLimits[joint]
            value = (range * action[joint])
            actual_joint_values.append(value)

        return actual_joint_values

    # Normalises the joint angles in radians to a value between [-1..1]
    def normaliseJointAngles(self, joint_positions):

        normalised_values = []
        # Check number of joint inputs is correct
        if len(joint_positions) != p.getNumJoints(self._kuka):
            return False

        # Check inputs are within maximum and minimum joint limits and amend if necessary
        lowerLimits, upperLimits = self.getKukaJointLimits
        for joint in np.arange(p.getNumJoints(self._kuka)):
            normal_value = (joint_positions[joint] / np.abs(upperLimits[joint]))

            # Cap to limits in-case of rounding error
            if normal_value > 1.0:
                normal_value = 1.0
            elif normal_value < -1.0:
                normal_value = -1.0

            normalised_values.append(normal_value)

        return normalised_values

    # Normalises the joint velocities to a value between [0..1]
    def normaliseJointVelocities(self, joint_velocities):

        normalised_values = []
        # Check number of joint inputs is correct
        if len(joint_velocities) != p.getNumJoints(self._kuka):
            return False

        # Check inputs are within maximum and minimum joint limits and amend if necessary
        lowerLimits, upperLimits = self.getKukaVelocityLimits
        for joint in np.arange(p.getNumJoints(self._kuka)):
            normal_value = (joint_velocities[joint] / np.abs(upperLimits[joint])) * 0.5
            normalised_values.append(0.5 + normal_value)

        return normalised_values


    # Normalise image values from [0..255] to [0..1]
    def normaliseImageValues(self, image):
        image = image.astype(dtype=np.float32)
        image /= 255.0
        return image

    # Advance simulation and collect observation, reward, and termination data
    def _step(self, action):

        self.action = action
        if self.terminated:
            # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()

        # Set target position to move towards
        target_joint_position = self.normalisedAction2JointAngles(self.action)
        self.setKukaJointAngles(target_joint_position)

        # Advance Simulation by 1 Environment timestep (by advancing through X SIMULATION_STEPS_PER_TIMESTEP)
        for step in np.arange(0, SIMULATION_STEPS_PER_TIMESTEP):
            p.stepSimulation()

        self._timestep_count += 1

        #todo check that reward and termination are in right order and doing the right things
        self._termination()
        self._observation = self.getObservation()
        self.reward = self._reward(self.reward_goal, self.reward_collision, self.reward_time, self.reward_rotation, self.reward_position)

        #todo check that these returns are correct re: reward and discount
        if self.terminated:
            return ts.termination(observation=self._observation, reward=self.reward)
        else:
            return ts.transition(observation=self._observation, reward=self.reward, discount=self.discount)

    def render(self, mode='human'):

        if mode=='human':
            print('Timestep: ' + str(self._timestep_count))
            print('Goal Pos: ' + str(self.target_position))
            print('End Effector Position: ' + str(self.effector_position))
            print('End Effector Orientation: ' + str(self.effector_orientation))
            print('Input Action: ' + str(self.action))
            print('Reward: ' + str(self.reward))
            print('Terminate: ' + str(self.terminated))
            print()
            #todo put in print statement for distance from goal
            #print(effector_position)
            #print(self.target_position)
            #print('Distance from Target Position: ' + str(np.abs(effector_position - self.target_position)))

        elif mode=='image':
            return self.get_images()

    # Fetches EIH and / or ETH images if enabled in environment
    def get_images(self):
        images = []
        if self._eih_input and self._eth_input:
            images.append(self.eih_rgba)
        if self._eth_input:
            images.append(self.eth_rgba)
        return images


    # Checks if a condition for termination has been met
    # Termination can be triggered by collision, goal achievement, or when the maximum number of steps have been reached
    def _termination(self):
        if self.terminated or self.goal_achieved or self._timestep_count > self.max_steps:
            self.terminated = True
            return True
        return False

    # Reward Function has 5 possible subfunctions which are incorporated depending on sparseness of reward signal
    # These subfunctions can be activated by setting the following flags
    # reward_goal: Binary reward signal of whether goal achieved or not
    # reward_collision: Binary reward signal of whether collision detected
    # reward_time: Incremental penalty for each timestep within allowed timelimit
    # reward_position: Incremental Reward for position towards target position
    # reward_rotation: Incremental penalty for deviation from target orientation

    def _reward(self, reward_goal=True, reward_collision=True, reward_time=True, reward_rotation=False, reward_position=False):

        reward = 0.0

        # 1) Calculate Kuka End Effector position and orientation (link 7)
        self.effector_position, self.effector_orientation, _, _, _, _ = p.getLinkState(self._kuka, 6,computeForwardKinematics=True)

        # 2) Retrieve Target Position
        target_position, target_orientation = p.getBasePositionAndOrientation(self.target_id)
        target_position = np.asarray(target_position)
        target_position[2] = target_position[2] + self.vertical_distance

        # 3) Calculate Euclidean distance between end effector and target
        distance = np.linalg.norm(target_position - np.asarray(self.effector_position))

        # 4) Calculate orientation matrix between effector orientation and target orientation
        target_orientation = np.array(p.getEulerFromQuaternion(target_orientation))
        target_orientation[0] += np.deg2rad(90)
        target_orientation = p.getQuaternionFromEuler(target_orientation)
        orientation_diff = p.getDifferenceQuaternion(target_orientation, self.effector_orientation)

        # 4.5 Calculate Magnitude of Orientation difference
        rotation = np.linalg.norm(orientation_diff)

        if reward_goal:

            # 5) Check if within dimensional tolerances, if true then goal achieved
            if (np.abs(distance) <= self.positional_tolerance) and (np.abs(rotation) <= self.rotational_tolerance):
                print('Goal Achieved')
                reward += MAX_REWARD_MAGNITUDE
                self.goal_achieved = True

            # ? 6) Verify that target is in camera?

        if reward_position and not self.goal_achieved:

            # 4) Return a fraction of maximum reward until within tolerance range
            max_positional_distance = (2 * KUKA_MAX_RANGE) - self.positional_tolerance
            position_reward = POSITION_REWARD_CONSTANT * MAX_REWARD_MAGNITUDE

            # if within tolerance, reward maximum allowed
            if np.abs(distance) <= self.positional_tolerance:
                reward = reward + position_reward
            # Otherwise, reward  a fraction depending on relative positional error
            else:
                reward = reward + position_reward * ((max_positional_distance - np.abs(distance)) / max_positional_distance)

        if reward_rotation and not self.goal_achieved:

            # 5) Return up to 0.4 of maximum reward until within tolerance range
            max_rotational_distance = 180 - self.rotational_tolerance
            rotation_reward = ROTATION_REWARD_CONSTANT * MAX_REWARD_MAGNITUDE

            # if within tolerance, reward maximum allowed
            if np.abs(rotation) <= self.rotational_tolerance:
                reward = reward + rotation_reward
            # Otherwise, reward  a fraction depending on relative rotational error
            else:
                reward = reward + rotation_reward * ((max_rotational_distance - np.abs(rotation)) / max_rotational_distance)

        #todo decide which reward function to use
        if reward_time and not self.goal_achieved:
            # Method 1) Reduce reward by a fixed amount every timestep
            reward = reward - TIME_PENALTY

        #if reward_time and self.terminated:
            # Method 2) Reduce maximum reward by an amount proportional to the number of timesteps
            # reward = reward * ((MAX_REWARD_MAGNITUDE / self.max_steps) * self._timestep_count)

        if reward_collision:
            # 1) Check if collision detected
            contact = p.getContactPoints(self._kuka, self.floor_id)
            if contact:
                # 2) if detected then end simulation and return maximum penalty
                print('Collision Detected')
                reward = -MAX_REWARD_MAGNITUDE
                self.terminated = True

        return reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]


    # Generate random coordinates for the episodic target and place as a flat urdf model in environment
    def generateGoal(self, x=None, y=None, size=1):

        # todo check x,y are valid
        if x and y:
            self.target_position = [x, y]
        else:
            #todo link random numbers to random seed
            # Generate 4 random numbers to choose 2 for the x and y coordinates of the target
            # This is to avoid the robots base and ensure targets are within robot's reach
            ranges = np.random.uniform(KUKA_MIN_RANGE, KUKA_MAX_RANGE, 4)
            x_ranges = np.stack((ranges[0], -ranges[1]))
            y_ranges = np.stack((ranges[2], -ranges[3]))
            target_x = np.random.choice(x_ranges)
            target_y = np.random.choice(y_ranges)
            self.target_position = [target_x, target_y]

        # Load URDF file and colour
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName='target.obj',
            rgbaColor=[1, 0, 0, 1],  # red
            meshScale=[size, size, size])

        # Place object at generated coordinates
        target_id = p.createMultiBody(
            baseVisualShapeIndex=visualShapeId,
            basePosition=[target_x, target_y, 0.0001],
            baseOrientation=p.getQuaternionFromEuler([np.deg2rad(90), 0, 0]))

        return target_id

    # Load floor urdf file and place in environment
    def generateFloor(self, size=1):

        # Load URDF file and colour
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName='floor.obj',
            rgbaColor=[0.6, 0.6, 0.6, 1],  # grey
            meshScale=[size, size, size])

        # Ensure floor can be collided with
        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName='floor.obj',
            meshScale=[size, size, size])

        # Merge collision and visual profile and place object
        floor = p.createMultiBody(
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([np.deg2rad(90), 0, 0]))

        return floor

    def generateRobot(self, jointPositions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        # Load KUKA iiwa Model
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self._kuka = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation)

        # Set KUKA Joint angles
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
        eih_dep = self._eih_channels
        eth_res = self._eth_camera_resolution
        eth_dep = self._eth_channels

        observation_spec: array_spec.ArraySpec = {}

        if self._eih_input:
            observation_spec.update({'eih_image': array_spec.BoundedArraySpec(shape=(eih_dep, eih_res, eih_res), dtype=np.float32, minimum=0,maximum=255)})

        
        if self._eth_input:
            observation_spec.update({'eth_image': array_spec.BoundedArraySpec(shape=(eth_dep, eth_res, eth_res), dtype=np.float32, minimum=0,maximum=255)})
            

        if self._position_input:
            observation_spec.update({'joint_positions': array_spec.BoundedArraySpec(shape=(7,), dtype=np.float32,
                                                                                    minimum=lower_limits,
                                                                                    maximum=upper_limits)})
        # todo could joint_velocities just be converted to an array spec if we are not using the bounds?
        if self._velocity_input:
            observation_spec.update({'joint_velocities': array_spec.BoundedArraySpec(shape=(7,), dtype=np.float32)})
                                                                                     #minimum=minimum_velocities,
                                                                                  #maximum=maximum_velocities)})
        return observation_spec

    def action_spec(self):
        """Return action_spec.
        Action spec consists of normalised values for the desired joint positions for the next time step."""

        return array_spec.BoundedArraySpec(shape=(7,), dtype=np.float32, minimum=-1.0, maximum=1.0, name='joint positions')
