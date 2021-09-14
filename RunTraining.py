from comet_ml import Experiment
from comet_ml import Optimizer
import base64
import imageio
import IPython
import matplotlib.pyplot as plt
import os
import tempfile
import reverb
import PIL.Image

import tf_agents.replay_buffers.table

import tensorflow as tf
import numpy as np

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

# Custom Env imports
from tf_agents.environments.utils import validate_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment

# Custom Network Imports
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tf_agents.networks.value_network import ValueNetwork

from KukaHybridVisualServoingEnv import KukaHybridVisualServoingEnv
from datetime import datetime

# Initialise Comet ML Logging
experiment = Experiment(
    api_key="HuSYxK8BJTK8MbTWpZIvnh91i",
    project_name="kukahybridvs",
    workspace="alexjameswilliams",
    log_code=True
)

tf.compat.v1.enable_v2_behavior()

tempdir = tempfile.gettempdir()
HD_IMAGE_RESOLUTION = 180
SD_IMAGE_RESOLUTION = 100
MONO = 1
RGB = 3
env = None


def train_new_model(hyperparameters, experiment):
    #############################################################################################
    #################### Initialise Environment #################################################
    #############################################################################################

    # Initialise Environment
    collect_env = KukaHybridVisualServoingEnv(renders=False, isDiscrete=False,
                                              eih_input=hyperparameters['eih_input'],
                                              eth_input=hyperparameters['eth_input'],
                                              position_input=hyperparameters['position_input'],
                                              velocity_input=hyperparameters['velocity_input'],
                                              eih_camera_resolution=hyperparameters['eih_resolution'],
                                              eth_camera_resolution=hyperparameters['eth_resolution'],
                                              eth_channels=hyperparameters['eth_channels'],
                                              eih_channels=hyperparameters['eih_channels'],
                                              timesteps=hyperparameters['collect_steps_per_iteration'],
                                              seed=hyperparameters['seed'])

    eval_env = KukaHybridVisualServoingEnv(renders=False, isDiscrete=False,
                                           eih_input=hyperparameters['eih_input'],
                                           eth_input=hyperparameters['eth_input'],
                                           position_input=hyperparameters['position_input'],
                                           velocity_input=hyperparameters['velocity_input'],
                                           eih_camera_resolution=hyperparameters['eih_resolution'],
                                           eth_camera_resolution=hyperparameters['eth_resolution'],
                                           eth_channels=hyperparameters['eth_channels'],
                                           eih_channels=hyperparameters['eih_channels'],
                                           timesteps=hyperparameters['collect_steps_per_iteration'],
                                           seed=hyperparameters['seed'])

    # Verify Environment specs work as intended
    print('Validating Environments')
    validate_py_environment(collect_env, episodes=1)
    validate_py_environment(eval_env, episodes=1)

    tf_collect_env = tf_py_environment.TFPyEnvironment(collect_env)
    tf_eval_env = tf_py_environment.TFPyEnvironment(eval_env)

    # time_limit_env = tf_agents.environments.wrappers.TimeLimit(env, )

    # Seed Environment (inspired by https://medium.com/@ODSC/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752)
    collect_env.seed(seed=hyperparameters['seed'])
    eval_env.seed(seed=hyperparameters['seed'])

    os.environ['PYTHONHASHSEED'] = str(hyperparameters['seed'])
    np.random.seed(hyperparameters['seed'])
    tf.random.set_seed(hyperparameters['seed'])

    ####################################
    ####### Picking a Strategy #########
    ####################################

    use_gpu = True  # @param {type:"boolean"}

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

    #############################################################################################
    #################### Initialise Network #####################################################
    #############################################################################################

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

    preprocessing_layers = {
        'eih_image': tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, 3),
                                                 tf.keras.layers.Flatten()]),
        'eth_image': tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, 3),
                                                 tf.keras.layers.Flatten()]),
        'joint_positions': tf.keras.layers.Dense(7),
        'joint_velocities': tf.keras.layers.Dense(7)
    }

    preprocessing_layers_critic = (preprocessing_layers,
                                   tf.keras.layers.Dense(7)
                                   )

    # Add each relevant input to network
    # preprocessing_layers = {}
    # if eih_input:
    #    preprocessing_layers.update({'eih_image': eih_img_pre_processing_layers})

    # if eth_input:
    #    preprocessing_layers.update({'eth_image': eth_img_pre_processing_layers})

    # if position_input:
    # preprocessing_layers.update({'joint_positions': joints_pre_processing_layers})

    # if velocity_input:
    # preprocessing_layers.update({'joint_velocities': velocities_pre_processing_layers})

    # Combine all Network Inputs Together
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(tf_collect_env)

    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=(256, 64),
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork
            )
        )

    with strategy.scope():
        critic_net = ValueNetwork(input_tensor_spec=(observation_spec, action_spec),
                                  preprocessing_layers=preprocessing_layers_critic,
                                  preprocessing_combiner=preprocessing_combiner,
                                  fc_layer_params=(256, 64),
                                  activation_fn='relu',
                                  kernel_initializer='random_normal')

    # Print network parameters / summary
    # actor_net.create_variables(input_tensor_spec=observation_spec)
    # actor_net.summary()
    # critic_net.create_variables(input_tensor_spec=observation_spec)
    # critic_net.summary()

    #############################################################################################
    #################### Initialise Agent #######################################################
    #############################################################################################

    with strategy.scope():
        train_step = train_utils.create_train_step()

        tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=hyperparameters['actor_learning_rate']),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=hyperparameters['critic_learning_rate']),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=hyperparameters['alpha_learning_rate']),
            target_update_tau=hyperparameters['target_update_tau'],
            target_update_period=hyperparameters['target_update_period'],
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=hyperparameters['gamma'],
            reward_scale_factor=hyperparameters['reward_scale_factor'],
            train_step_counter=train_step)

        tf_agent.initialize()

    #############################################################################################
    #################### Replay Buffer ##########################################################
    #############################################################################################

    rate_limiter = reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3,
                                                            error_buffer=3.0)

    table_name = 'uniform_table'
    table = reverb.Table(
        table_name,
        max_size=hyperparameters['replay_buffer_capacity'],
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))

    reverb_server = reverb.Server([table])

    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        sequence_length=2,
        table_name=table_name,
        local_server=reverb_server)

    dataset = reverb_replay.as_dataset(
        sample_batch_size=hyperparameters['batch_size'], num_steps=2).prefetch(50)
    experience_dataset_fn = lambda: dataset

    #############################################
    ##########Initialise Policies################
    #############################################

    # Initialise 3 policies for collecting experience and evaluation

    # Main policy used for evaluation and deployment.
    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_eval_policy, use_tf_function=True)

    # Policy used for exploration and data collection
    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy, use_tf_function=True)

    # Policy that randomly selects an action for each timestep for random exploration
    random_policy = random_py_policy.RandomPyPolicy(collect_env.time_step_spec(), collect_env.action_spec())

    #############################################
    ##########Initialise Actors##################
    #############################################

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        reverb_replay.py_client,
        table_name,
        sequence_length=2,
        stride_length=1)

    # Instantiate 3 Actors to manage interactions between policies and an environment

    # Create an Actor with the random policy to collect experiences to seed the replay buffer with.
    initial_collect_actor = actor.Actor(
        collect_env,
        random_policy,
        train_step,
        steps_per_run=hyperparameters['initial_collect_steps'],
        observers=[rb_observer])
    initial_collect_actor.run()

    # Instantiate an Actor with the collect policy to gather more experiences during training.
    env_step_metric = py_metrics.EnvironmentSteps()
    collect_actor = actor.Actor(
        collect_env,
        collect_policy,
        train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(10),
        summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
        observers=[rb_observer, env_step_metric])

    # Create an Actor which will be used to evaluate the policy during training.
    eval_actor = actor.Actor(
        eval_env,
        eval_policy,
        train_step,
        episodes_per_run=hyperparameters['num_eval_episodes'],
        metrics=actor.eval_metrics(hyperparameters['num_eval_episodes']),
        summary_dir=os.path.join(tempdir, 'eval'),
    )

    ##################################
    #########Instantiate Learner######
    ##################################
    # The Learner component contains the agent and performs gradient step updates
    # to the policy variables using experience data from the replay buffer.

    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            tf_agent,
            train_step,
            interval=hyperparameters['policy_save_interval']),
        triggers.StepPerSecondLogTrigger(train_step, interval=1000),
    ]

    agent_learner = learner.Learner(
        tempdir,
        train_step,
        tf_agent,
        experience_dataset_fn,
        triggers=learning_triggers,
        run_optimizer_variable_init=False)

    ######################################
    ######## Evaluation Metrics ##########
    ######################################

    # The eval actor has been instantiatied with the most commonly used metrics:
    # average return; average episode length;
    # Metrics are generated by running this actor
    def get_eval_metrics():
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
        return results

    metrics = get_eval_metrics()

    # todo add comet.ml logging
    def log_eval_metrics(step, metrics):
        eval_results = (', ').join(
            '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('step = {0}: {1}'.format(step, eval_results))
        # experiment.log_metric("loss",loss_val,step=step)

    ######################################
    ############# Begin Training##########
    ######################################

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]

    for _ in range(hyperparameters['num_iterations']):
        # Training.
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)

        # Evaluating.
        step = agent_learner.train_step_numpy
        experiment.set_step(step)
        print('Step: ' + str(step))

        if hyperparameters['eval_interval'] and step % hyperparameters['eval_interval'] == 0:
            metrics = get_eval_metrics()
            log_eval_metrics(step, metrics)
            experiment.log_metrics(metrics)
            returns.append(metrics["AverageReturn"])

        if hyperparameters['log_interval'] and step % hyperparameters['log_interval'] == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]


          experiment.set_step(step)
          metrics = get_eval_metrics()
          returns.append(metrics["AverageReturn"])
          log_eval_metrics(metrics)


          video_file_name = record_video()
          experiment.log_asset(video_file_name, step=None, overwrite=None, context=None, ftype='video', metadata=None)


    video_file_name = record_video(eval_actor, eval_env, video_name = 'final')
    experiment.log_asset(video_file_name, step=None, overwrite=None, context=None, ftype='video', metadata=None)

    # Closer reverb server before exiting
    rb_observer.close()
    reverb_server.stop()
    collect_env.close()
    eval_env.close()

    return avg_return

    # Terminate Comet ML Logging
    # experiment.end()


# steps = range(0, num_iterations + 1, eval_interval)
# plt.plot(steps, returns)
# plt.ylabel('Average Return')
# plt.xlabel('Step')
# plt.ylim()


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)

# Render a video
def record_video(actor, env, video_name=None):

  video_episodes = 5
  eih_video_filename = 'eih.mp4'
  eth_video_filename = 'eth.mp4'
  with imageio.get_writer(eih_video_filename, fps=60) as eih_video:
      with imageio.get_writer(eth_video_filename, fps=60) as eth_video:
          for _ in range(video_episodes):
              time_step = env.reset()

              images = env.get_images()
              eih_video.append_data(images[0])
              eth_video.append_data(images[1])

              while not time_step.is_last():
                  action_step = actor.policy.action(time_step)
                  time_step = env.step(action_step.action)

                  images = env.get_images()
                  eih_video.append_data(images[0])
                  eth_video.append_data(images[1])

  embed_mp4(eih_video_filename)
  embed_mp4(eth_video_filename)

  if video_name is None:
    video_name = datetime.now().strftime("%Y-%m-%d---%H-%M-%S")

  video_name = (video_name + ".mp4")
  # Combine videos into one side by side video
  os.system("ffmpeg -i eih.mp4 -i eth.mp4 -filter_complex \"hstack,format=yuv420p\" -c:v libx264 -crf 18 " + video_name)

  return video_name


def run_single_experiment():
    experiment = Experiment(api_key="HuSYxK8BJTK8MbTWpZIvnh91i",
                            project_name="kukahybridvs",
                            workspace="alexjameswilliams",
                            log_code=True)

    # Set Default Hyperparameters
    # Training Hyperparameters
    num_iterations = 100000
    initial_collect_steps = 10000  # @param {type:"integer"}
    collect_steps_per_iteration = 20  # @param {type:"integer"}
    num_eval_episodes = 20  # @param {type:"integer"}
    eval_interval = 10000  # @param {type:"integer"}
    policy_save_interval = 5000  # @param {type:"integer"}
    log_interval = 5000  # @param {type:"integer"}
    replay_buffer_capacity = 10000  # @param {type:"integer"}
    batch_size = 256
    seed = 1234567

    # RL Hyperparameters
    critic_learning_rate = 3e-4  # @param {type:"number"}
    actor_learning_rate = 3e-4  # @param {type:"number"}
    alpha_learning_rate = 3e-4  # @param {type:"number"}
    target_update_tau = 0.005  # @param {type:"number"}
    target_update_period = 1  # @param {type:"number"}
    gamma = 0.99  # @param {type:"number"}
    reward_scale_factor = 1.0  # @param {type:"number"}

    # Environment Hyperparameters
    eih_input = True  # Set to true to include input from EIH Camera in environment / network
    eth_input = True  # Set to true to include input from ETH Camera in environment / network
    position_input = True  # Set to true to include input from position sensors in environment / network
    velocity_input = True  # Set to true to include input from velocity sensors in environment / network
    combined_actor_critic = False
    eih_resolution = SD_IMAGE_RESOLUTION
    eth_resolution = SD_IMAGE_RESOLUTION
    eih_channels = RGB
    eth_channels = RGB

    # Model Hyperparameters
    hidden_activation = tf.keras.activations.relu
    hidden_kernel_initializer = tf.keras.initializers.random_normal
    hidden_bias_initializer = tf.keras.initializers.zeros
    output_activation = tf.keras.activations.tanh
    output_kernal_initializer = tf.keras.initializers.random_normal
    output_bias_initializer = tf.keras.initializers.zeros
    mlp1_node_params = [32, 64]  # Format: [l1_nodes,....,ln_nodes]
    mlp2_node_params = [256, 64]  # Format: [l1_nodes,....,ln_nodes]
    cnn_node_params = [[32, 3], [32, 3]]  # Format: [[l1_filters,l1_kernel_size]...[ln_filters,ln_kernel_size]]

    # Get Hyperparameters for this experimental run
    hyperparameters = {
        'num_iterations': num_iterations,
        'initial_collect_steps': initial_collect_steps,
        'collect_steps_per_iteration': collect_steps_per_iteration,
        'num_eval_episodes': num_eval_episodes,
        'eval_interval': eval_interval,
        'policy_save_interval': policy_save_interval,
        'log_interval': log_interval,
        'replay_buffer_capacity': replay_buffer_capacity,
        'batch_size': batch_size,
        'seed': seed,

        # RL Hyperparameters
        'critic_learning_rate': critic_learning_rate,
        'actor_learning_rate': actor_learning_rate,
        'alpha_learning_rate': alpha_learning_rate,
        'target_update_tau': target_update_tau,
        'target_update_period': target_update_period,
        'gamma': gamma,
        'reward_scale_factor': reward_scale_factor,

        # Environment Hyperparameters
        'eih_input': eih_input,
        'eth_input': eth_input,
        'position_input': position_input,
        'velocity_input': velocity_input,
        'combined_actor_critic': combined_actor_critic,
        'eih_resolution': eih_resolution,
        'eth_resolution': eth_resolution,
        'eih_channels': eih_channels,
        'eth_channels': eth_channels,

        # Model Hyperparameters
        'hidden_activation': hidden_activation,
        'hidden_kernel_initializer': hidden_kernel_initializer,
        'hidden_bias_initializer': hidden_bias_initializer,
        'output_activation': output_activation,
        'output_kernal_initializer': output_kernal_initializer,
        'output_bias_initializer': output_bias_initializer,
        'mlp1_node_params': mlp1_node_params,
        'mlp2_node_param': mlp2_node_params,
        'cnn_node_params': cnn_node_params
    }

    experiment.log_parameters(hyperparameters)
    train_new_model(hyperparameters, experiment)

    experiment.end()


def run_experiments():
    # Set Default Hyperparameters
    # Training Hyperparameters
    num_iterations = 100000
    initial_collect_steps = 10000  # @param {type:"integer"}
    collect_steps_per_iteration = 20  # @param {type:"integer"}
    num_eval_episodes = 20  # @param {type:"integer"}
    eval_interval = 10000  # @param {type:"integer"}
    policy_save_interval = 5000  # @param {type:"integer"}
    log_interval = 5000  # @param {type:"integer"}
    replay_buffer_capacity = 10000  # @param {type:"integer"}
    batch_size = 256
    seed = 1234567

    # RL Hyperparameters
    critic_learning_rate = 3e-4  # @param {type:"number"}
    actor_learning_rate = 3e-4  # @param {type:"number"}
    alpha_learning_rate = 3e-4  # @param {type:"number"}
    target_update_tau = 0.005  # @param {type:"number"}
    target_update_period = 1  # @param {type:"number"}
    gamma = 0.99  # @param {type:"number"}
    reward_scale_factor = 1.0  # @param {type:"number"}

    # Environment Hyperparameters
    eih_input = True  # Set to true to include input from EIH Camera in environment / network
    eth_input = True  # Set to true to include input from ETH Camera in environment / network
    position_input = True  # Set to true to include input from position sensors in environment / network
    velocity_input = True  # Set to true to include input from velocity sensors in environment / network
    combined_actor_critic = False
    eih_resolution = SD_IMAGE_RESOLUTION
    eth_resolution = SD_IMAGE_RESOLUTION
    eih_channels = RGB
    eth_channels = RGB

    # Model Hyperparameters
    hidden_activation = tf.keras.activations.relu
    hidden_kernel_initializer = tf.keras.initializers.random_normal
    hidden_bias_initializer = tf.keras.initializers.zeros
    output_activation = tf.keras.activations.tanh
    output_kernal_initializer = tf.keras.initializers.random_normal
    output_bias_initializer = tf.keras.initializers.zeros
    mlp1_node_params = [32, 64]  # Format: [l1_nodes,....,ln_nodes]
    mlp2_node_params = [256, 64]  # Format: [l1_nodes,....,ln_nodes]
    cnn_node_params = [[32, 3], [32, 3]]  # Format: [[l1_filters,l1_kernel_size]...[ln_filters,ln_kernel_size]]

    # Specify Random search Parameters and optimiser algorithm
    config = {

        "name": "rl-hyperparameters-optimisation-1",
        "algorithm": "bayes",

        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "critic_learning_rate": {"type": "float", "min": 3e-5, "max": 3e-2, "scalingType": "loguniform"},
            "actor_learning_rate": {"type": "float", "min": 3e-5, "max": 3e-2, "scalingType": "loguniform"},
            "alpha_learning_rate": {"type": "float", "min": 0.00001, "max": 0.1, "scalingType": "loguniform"},
            "target_update_tau": {"type": "float", "min": 0.0, "max": 0.1, "scalingType": "loguniform"},
            "gamma": {"type": "float", "min": 0.9, "max": 0.999, "scalingType": "loguniform"},
        },

        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "AverageReturn",
            "objective": "maximize",
            "seed": 1
        },
    }

    # Instantiate Optimiser / Experiment
    opt = Optimizer(config, trials=1,
                    api_key="HuSYxK8BJTK8MbTWpZIvnh91i",
                    project_name="kukahybridvs",
                    workspace="alexjameswilliams",
                    log_code=True)

    for experiment in opt.get_experiments():
        experiment.add_tag("rl-hyperparameters-optimisation-1")
        # Call the function that wraps the Neural Network code to start the experiment

        # Get Hyperparameters for this experimental run
        hyperparameters = {
            'num_iterations': num_iterations,
            'initial_collect_steps': initial_collect_steps,
            'collect_steps_per_iteration': collect_steps_per_iteration,
            'num_eval_episodes': num_eval_episodes,
            'eval_interval': eval_interval,
            'policy_save_interval': policy_save_interval,
            'log_interval': log_interval,
            'replay_buffer_capacity': replay_buffer_capacity,
            'batch_size': batch_size,
            'seed': seed,

            # RL Hyperparameters
            'critic_learning_rate': experiment.get_parameter('critic_learning_rate'),
            'actor_learning_rate': experiment.get_parameter('actor_learning_rate'),
            'alpha_learning_rate': experiment.get_parameter('alpha_learning_rate'),
            'target_update_tau': experiment.get_parameter('target_update_tau'),
            'target_update_period': target_update_period,
            'gamma': experiment.get_parameter('gamma'),
            'reward_scale_factor': reward_scale_factor,

            # Environment Hyperparameters
            'eih_input': eih_input,
            'eth_input': eth_input,
            'position_input': position_input,
            'velocity_input': velocity_input,
            'combined_actor_critic': combined_actor_critic,
            'eih_resolution': eih_resolution,
            'eth_resolution': eth_resolution,
            'eih_channels': eih_channels,
            'eth_channels': eth_channels,

            # Model Hyperparameters
            'hidden_activation': hidden_activation,
            'hidden_kernel_initializer': hidden_kernel_initializer,
            'hidden_bias_initializer': hidden_bias_initializer,
            'output_activation': output_activation,
            'output_kernal_initializer': output_kernal_initializer,
            'output_bias_initializer': output_bias_initializer,
            'mlp1_node_params': mlp1_node_params,
            'mlp2_node_param': mlp2_node_params,
            'cnn_node_params': cnn_node_params
        }

        experiment.log_parameters(hyperparameters)

        print('Beginning Experiment')
        print(hyperparameters)

        # Start Experiments
        avg_return = train_new_model(hyperparameters, experiment)
        experiment.log_metric("AverageReturn", avg_return)


if __name__ == "__main__":
    # run_experiments()
    run_single_experiment()