# A reinforcement learning environment for the Kuka LBR iiwa with Hybrid Visual Servoing Input
A customisable simulated reinforcement learning Environment for learning accurate manipulation over 2D surfaces with the KUKA LBR iiwa via a Hybrid Visual Servoing method.
The hybrid visual servoing approach includes eye-in-hand and eye-to-hand view inputs, as well as robotic position and velocity inputs.

Simulation is developed in PyBullet, with TFAgents used as the wrapper for the environment logic for reinforcement learning algorithms.
An additional script for initialising and commencing training on the environment is provided to help users get started.

Please cite this thesis if you use this work in your projects:

_A. Williams, “Real-Time Hybrid Visual Servoing of a Redundant Manipulator via Deep Reinforcement Learning,” MSc Thesis, Swansea University, Swansea, 2021. Available: https://cronfa.swan.ac.uk/Record/cronfa62598_
