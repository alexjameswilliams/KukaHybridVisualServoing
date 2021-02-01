import tensorflow as tf
import numpy as np
import tf_agents
from tensorflow import keras
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments import utils
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import random_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
tf.compat.v1.enable_v2_behavior()
HD_IMAGE_RESOLUTION = 180
SD_IMAGE_RESOLUTION = 120
MONO = 1
RGB = 3
NORMALIZED_INPUT_RANGE = 5
env = None
def main():
    collect_steps_per_iteration = 20  # @param {type:"integer"}
    eih_resolution = SD_IMAGE_RESOLUTION,
    eth_resolution = SD_IMAGE_RESOLUTION,
    img_channels = RGB,
    # Initialise Environment
    environment = KukaHybridVisualServoingEnv(renders=True, isDiscrete=False, eih_camera_resolution=eih_resolution,
                                              eth_camera_resolution=eth_resolution, image_depth=img_channels,
                                              steps=collect_steps_per_iteration)

    # Add joint parameter controls (for manual debugging)
    lowerLimits, upperLimits = environment.getKukaJointLimits
    controlIds = []
    for jointIndex in range(len(lowerLimits)):
        controlIds.append(
            environment._p.addUserDebugParameter("A", np.rad2deg(lowerLimits[jointIndex]), -np.rad2deg(lowerLimits[jointIndex]),
                                         0))
    # Verify Environment specs work as intended
    tf_agents.environments.utils.validate_py_environment(environment, episodes=1)

    tf_env = tf_py_environment.TFPyEnvironment(environment)

if __name__ == "__main__":
    main()
