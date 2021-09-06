import numpy as np
from KukaHybridVisualServoingEnv import KukaHybridVisualServoingEnv
from tf_agents.environments.utils import validate_py_environment
from tf_agents.environments import py_environment
from PIL import Image
import base64
import imageio
import IPython
import os
from io import BytesIO

def main():

    env = KukaHybridVisualServoingEnv(renders=True, isDiscrete=False, seed=0)

    validate_py_environment(env, episodes=1)

    # Add joint parameter controls
    lowerLimits, upperLimits = env.getKukaJointLimits
    controlIds = []
    for jointIndex in range(len(lowerLimits)):
        controlIds.append(env._p.addUserDebugParameter("A", np.rad2deg(lowerLimits[jointIndex]),-np.rad2deg(lowerLimits[jointIndex]),0))


    # Export 3 episodes for .mp4 videi
    num_episodes = 3
    eih_video_filename = 'eih.mp4'
    eth_video_filename = 'eth.mp4'
    with imageio.get_writer(eih_video_filename, fps=60) as eih_video:
        with imageio.get_writer(eth_video_filename, fps=60) as eth_video:
            for _ in range(num_episodes):
                time_step = env.reset()
                eih_video.append_data(env.get_images()[0])
                eth_video.append_data(env.get_images()[1])
                while not time_step.is_last():
                    action = []
                    for controlId in controlIds:
                        action.append(np.deg2rad(env._p.readUserDebugParameter(controlId)))

                    action = env.normaliseJointAngles(action)
                    time_step = env.step(action)
                    env.render()
                    eih_video.append_data(env.get_images()[0])
                    eth_video.append_data(env.get_images()[1])

    embed_mp4(eih_video_filename)
    embed_mp4(eth_video_filename)

    # Combine videos into one side by side video
    os.system("ffmpeg -i eih.mp4 -i eth.mp4 -filter_complex \"hstack,format=yuv420p\" -c:v libx264 -crf 18 output.mp4")


    done = False
    episode_count = 0
    timestep = env.reset()
    while (not done):

        if timestep.is_first():
            episode_count = episode_count + 1
            eih_imgs = []
            eth_imgs = []
        elif timestep.is_last():
            images = env.get_images()
            eih_im = Image.fromarray(images[0])
            eth_im = Image.fromarray(images[1])
            eih_im.save(str(episode_count) + "_EIH.png")
            eth_im.save(str(episode_count) + "_ETH.png")

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

def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)

if __name__ == "__main__":
    main()