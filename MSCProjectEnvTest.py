import numpy as np
from KukaHybridVisualSevoingEnv import KukaHybridVisualSevoingEnv


def main():
    env = KukaHybridVisualSevoingEnv(renders=True, isDiscrete=False)

    # Add joint parameter controls
    lowerLimits, upperLimits = env.getKukaJointLimits
    motorsIds = []
    for jointIndex in range(len(lowerLimits)):
        motorsIds.append(env._p.addUserDebugParameter("A", np.rad2deg(lowerLimits[jointIndex]),-np.rad2deg(lowerLimits[jointIndex]),0))

    done = False
    while (not done):

        action = []
        for motorId in motorsIds:
            action.append(np.deg2rad(env._p.readUserDebugParameter(motorId)))

        env.step(action)
        #state, reward, done, info = environment.step(action)
        #obs = environment.getObservation()


def getKukaJointVelocityLimits():
    return np.array([98,98,100,130,140,180,180])


if __name__ == "__main__":
    main()