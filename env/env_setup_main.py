from env.env_setup import PushTImageEnv
import numpy as np


def setup_env():
    print("Setting up environment...")
    env = PushTImageEnv()
    env.seed(1000)

    # reset
    obs, info = env.reset()

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    with np.printoptions(precision=4, suppress=True, threshold=5):
        print("obs['image'].shape:", obs['image'].shape, "float32, [0,1]")
        print("obs['agent_pos'].shape:",
              obs['agent_pos'].shape, "float32, [0,512]")
        print("action.shape: ", action.shape, "float32, [0,512]")

    print("Environment setup complete.")
    return env
