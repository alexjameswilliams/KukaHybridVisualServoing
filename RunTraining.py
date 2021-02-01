import tensorflow as tf
import numpy as np
import tf_agents
from tf_agents.environments.py_environment import PyEnvironment

from KukaHybridVisualServoingEnv import KukaHybridVisualServoingEnv

from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model

from tf_agents.networks import *
from tf_agents.networks.network import DistributionNetwork
from tf_agents.agents.ppo import ppo_agent

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
    # Model Hyperparameters
    eih_input = True,   # Set to true to include input from EIH Camera in environment / network
    eth_input = True,   # Set to true to include input from ETH Camera in environment / network
    position_input = True,  # Set to true to include input from position sensors in environment / network
    velocity_input = True,  # Set to true to include input from velocity sensors in environment / network
    eih_resolution = SD_IMAGE_RESOLUTION,
    eth_resolution = SD_IMAGE_RESOLUTION,
    img_channels = RGB,
    # PPO Hyperparameters
    importance_ratio_clipping = 0.2,
    lambda_value = 1.0,
    discount_factor = 0.995,
    num_epochs = 15,  # @param {type:"integer"} Number of epochs for computing policy updates. / Basically how many episodes before updating the policy I think
    use_gae = True,  # Use generalized advantage estimation for computing per-timestep advantage
    use_td_lambda_return = True,  # Use td_lambda_return for training value function
    normalize_rewards = True,  # If True, keeps moving variance of rewards and normalizes incoming rewards.
    reward_norm_clipping = 1,  # Value above and below to clip normalized reward.
    normalize_observations = True

    # Initialise Environment
    environment = KukaHybridVisualServoingEnv(renders=True, isDiscrete=False,
                                              eih_input=eih_input, eth_input=eth_input, position_input=position_input, velocity_input=velocity_input,
                                              eih_camera_resolution=eih_resolution, eth_camera_resolution=eth_resolution, image_depth=img_channels,
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

    # Image Inputs Network / CNN Block
    eih_img_pre_processing_layers = tf.keras.models.Sequential(
        [Conv2D(filters=32, kernel_size=(2, 2), activation='relu',
                bias_initializer='zeros', kernel_initializer='random_normal'),
         Conv2D(filters=32, kernel_size=(2, 2), activation='relu',
                bias_initializer='zeros', kernel_initializer='random_normal'),
         Flatten()])

    eth_img_pre_processing_layers = tf.keras.models.Sequential(
        [Conv2D(filters=32, kernel_size=(2, 2), activation='relu',
                bias_initializer='zeros', kernel_initializer='random_normal'),
         Conv2D(filters=32, kernel_size=(2, 2), activation='relu',
                bias_initializer='zeros', kernel_initializer='random_normal'),
         Flatten()])

    # Vector Inputs / MLP 1 Block
    joints_pre_processing_layers = tf.keras.models.Sequential(
        [Dense(32, activation='relu', bias_initializer='zeros', kernel_initializer='random_normal'),
         Dense(32, activation='relu', bias_initializer='zeros', kernel_initializer='random_normal')])

    velocities_pre_processing_layers = tf.keras.models.Sequential(
        [Dense(32, activation='relu', bias_initializer='zeros', kernel_initializer='random_normal'),
         Dense(32, activation='relu', bias_initializer='zeros', kernel_initializer='random_normal')])

    # Add each relevant input to network 
    preprocessing_layers = {}
    if eih_input:
        preprocessing_layers.update({'eih_image': eih_img_pre_processing_layers})

    if eth_input:
        preprocessing_layers.update({'eth_image': eth_img_pre_processing_layers})

    if position_input:
        preprocessing_layers.update({'joint_positions': joints_pre_processing_layers})

    if velocity_input:
        preprocessing_layers.update({'joint_velocities': velocities_pre_processing_layers})

    # Combine all Network Inputs Together
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    # Create Actor Network
    actor = actor_distribution_network.ActorDistributionNetwork(input_tensor_spec=tf_env.observation_spec(),
                                                                output_tensor_spec=tf_env.action_spec(),
                                                                preprocessing_layers=preprocessing_layers,
                                                                preprocessing_combiner=preprocessing_combiner,
                                                                fc_layer_params=(256, 64),
                                                                activation_fn='relu',
                                                                kernel_initializer='random_normal')

    # Create Critic Network
    critic = value_network.ValueNetwork(input_tensor_spec=tf_env.observation_spec(),
                                        preprocessing_layers=preprocessing_layers,
                                        preprocessing_combiner=preprocessing_combiner,
                                        fc_layer_params=(256, 64),
                                        activation_fn='relu',
                                        kernel_initializer='random_normal')

    actor.create_variables(input_tensor_spec=tf_env.observation_spec())
    actor.summary()
    critic.create_variables(input_tensor_spec=tf_env.observation_spec())
    critic.summary()

    # Initialise PPO Agent.
    # For more information see: https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/PPOAgent
    # PPO Tutorial: https://www.arconsis.com/unternehmen/blog/reinforcement-learning-doom-with-tf-agents-and-ppo
    tf_agent = ppo_agent.PPOAgent(
        time_step_spec=tf_env.time_step_spec(),  # A TimeStep spec of the environment.
        action_spec=tf_env.action_spec(),  # A nest of BoundedTensorSpec representing the actions.
        actor_net=actor,
        value_net=critic,
        importance_ratio_clipping=importance_ratio_clipping,
        lambda_value=lambda_value,
        discount_factor=discount_factor,
        num_epochs=num_epochs,  # Number of epochs for computing policy updates.
        use_gae=use_gae,  # Use generalized advantage estimation for computing per-timestep advantage
        use_td_lambda_return=use_td_lambda_return,  # Use td_lambda_return for training value function
        normalize_rewards=normalize_rewards,
        # If True, keeps moving variance of rewards and normalizes incoming rewards.
        reward_norm_clipping=reward_norm_clipping,  # Value above and below to clip normalized reward.
        normalize_observations=normalize_observations
        # If True, keeps moving mean and variance of observations and normalizes incoming observations.
    )
    tf_agent.initialize()
if __name__ == "__main__":
    main()
