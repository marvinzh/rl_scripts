import gymnasium as gym
from agent import PPOAgentProfile, AgentInitializer
from torchrl.envs import ParallelEnv, GymEnv


def load_envs(env_name, n_env):
    envs = ParallelEnv(n_env, lambda: GymEnv(env_name))
    return envs


def load_env(env_name, visual: bool):
    mode = "human" if visual else "rgb_array"
    env = gym.make(env_name, render_mode=mode)
    return env


def load_agent(profile_path):
    profile = PPOAgentProfile()
    profile.load(profile_path)
    profile.pprint()
    agent = AgentInitializer.create_agent(profile)
    return agent


def extract_space_size(space: gym.Space):
    if isinstance(space, gym.spaces.Box):
        return sum(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    else:
        raise NotImplementedError("Unknown type: %s" % type(space))
