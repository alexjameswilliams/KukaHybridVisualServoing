import numpy as np
from KukaHybridVisualServoingEnv import KukaHybridVisualServoingEnv
from tf_agents.environments.utils import validate_py_environment
from tf_agents.environments import py_environment


def main():

    env = KukaHybridVisualServoingEnv(renders=True, isDiscrete=False, seed=0)

    validate_py_environment(env, episodes=1)

    # Add joint parameter controls
    lowerLimits, upperLimits = env.getKukaJointLimits
    controlIds = []
    for jointIndex in range(len(lowerLimits)):
        controlIds.append(env._p.addUserDebugParameter("A", np.rad2deg(lowerLimits[jointIndex]),-np.rad2deg(lowerLimits[jointIndex]),0))


    done = False
    while (not done):

        action = []
        for controlId in controlIds:
            action.append(np.deg2rad(env._p.readUserDebugParameter(controlId)))


        action = env.normaliseJointAngles(action)
        timestep = env.step(action)
        env.render()

        if done:
            print('TERMINATED')


def getKukaJointVelocityLimits():
    return np.array([98,98,100,130,140,180,180])


if __name__ == "__main__":
    main()