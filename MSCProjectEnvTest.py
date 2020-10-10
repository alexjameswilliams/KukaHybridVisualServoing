import numpy as np
from KukaHybridVisualSevoingEnv import KukaHybridVisualSevoingEnv


def main():
    environment = KukaHybridVisualSevoingEnv(renders=True, isDiscrete=False)

    done = False
    while (not done):

        action = []
        for motorId in motorsIds:
            action.append(environment._p.readUserDebugParameter(motorId))

        #state, reward, done, info = environment.step(action)
        #obs = environment.getObservation()

if __name__ == "__main__":
    main()