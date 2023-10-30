import argparse
from argparse import Namespace
import time

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from utils.general import pprint_summary
from agent import PPOAgentProfile, AgentInitializer
from commons import load_envs, load_env, extract_space_size
from eval import play_a_round


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("agent", choices=["PPO"])
    parser.add_argument(
        "env", type=str, help="Environment name of OpenAI Gymnasium")
    parser.add_argument("--n_envs", "-ne", type=int, default=2,
                        help="# of Parallelized Environment for collecing trajatory")
    parser.add_argument("--optimizer", "-opt", type=str,
                        choices=["adam"], default="adam", help="")
    parser.add_argument("--gamma", type=float, default=.99,
                        help="discount factor [0, 1]")
    parser.add_argument("--lambda", type=float, default=.95,
                        help="lambda-return factor [0, 1]")
    parser.add_argument("--vf_coef", type=float,
                        help="weight of value function loss")
    parser.add_argument("--update_interval", type=int,
                        default=2048, help="model update interval in step")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batchsize for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="training epochs in an update")
    parser.add_argument("--clip_eps", type=float, default=.2,
                        help="Epsilon for pessimistic clipping of likelihood ratio up update policy")
    parser.add_argument("--clip_eps_vf", type=float, default=None,
                        help="Epsilon for pessimistic clipping of value to update value function. If it is `None`, value function is not clipped on updates")
    parser.add_argument("--std_advantages", type=bool, default=True,
                        help="Use standardized advantages on updates")
    parser.add_argument("--max_grad_norm", type=float, default=None,
                        help="Maximum L2 norm of the gradient used for gradient clipping. If set to None, the gradient is not clipped.")

    parser.add_argument("--learning_rate", "-lr", type=float,
                        default=1e-2, help="learning rate")
    parser.add_argument("--max_episode", type=int,
                        default=1e6, help="max episode")
    parser.add_argument("--policy_hiddens", nargs="+", default=[32], help="")
    parser.add_argument(
        "--policy_type", choices=["MLPWithSoftmaxHead"], default="MLPWithSoftmaxHead", type=str, help="")
    parser.add_argument(
        "--vf_type", choices=["MLP"], default="MLP", type=str, help="")
    parser.add_argument("--vf_hiddens", nargs="+", default=[64, 64], help="")
    parser.add_argument(
        "--vf_act", choices=["tanh"], default="tanh", type=str, help="")

    parser.add_argument("--evaluation_interval", type=int, default=10000, help="")

    return parser.parse_args()


def create_agent(args: Namespace, env: gym.Env):
    p = PPOAgentProfile()
    p.read_from_dict({
        "obs_size": extract_space_size(env.observation_space),
        "action_size": extract_space_size(env.action_space),
        "agent_type": args.agent,
        "policy_hiddens": args.policy_hiddens,
        "policy_type": args.policy_type,
        "vf_act": args.vf_act,
        "vf_hiddens": args.vf_hiddens,
        "vf_type": args.vf_type,
    })
    p.pprint()
    p.save("%s-%s-%s.json"%(args.agent, args.policy_type, args.env))
    agent = AgentInitializer.create_agent(p)
    return agent


def batch_act_and_observe_step(agent, envs, obs):
    action_size = envs.action_spec.space.n
    n_episode = 0
    actions = agent.batch_act(obs)
    td = envs.step(TensorDict({
        "action": torch.nn.functional.one_hot(torch.LongTensor(actions), action_size)
    }, envs.batch_size.numel()))
    next_obs = td["next"]["observation"]
    rewards, dones, truncateds = td["next"]["reward"], td["next"]["done"], td["next"]["truncated"]
    agent.batch_observe(next_obs, rewards.squeeze(),
                        dones.squeeze(), truncateds.squeeze())

    # reset environment if any of them reach terminate state
    if torch.any(torch.logical_or(dones, truncateds)):
        reset_mask = torch.logical_or(dones, truncateds)
        td = envs.reset(
            TensorDict({
                "_reset": reset_mask
            }, envs.batch_size.numel())
        )
        next_obs = td["observation"]
        n_episode += torch.where(reset_mask, 1, 0).sum().numel()

    return next_obs, n_episode


def validate_step(env, agent, n=10):
    Rs = []
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        R = 0
        i_step = 0
        while i_step < env.spec.max_episode_steps:
            i_step += 1
            with agent.eval_mode():
                action = agent.act(obs)
                obs, r, done, truncated, info = env.step(action)
                R += r

            if done or truncated:
                break
        Rs.append(R)

    return np.mean(Rs), np.std(Rs)


def main(args: Namespace):
    envs = load_envs(args.env, args.n_envs)
    env = load_env(args.env, False)
    env.reset()

    agent = create_agent(args, env)
    print(agent.model)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            agent.model.parameters(), args.learning_rate)

    agent.optimizer = optimizer

    td = envs.reset()
    obs = td["observation"]
    total_episode = 0
    sampling_step = 0
    best_score = None
    while total_episode < args.max_episode:
        sampling_step += 1
        obs, n_episode = batch_act_and_observe_step(agent, envs, obs)
        total_episode += n_episode
        statistics = agent.get_statistics()
        print("steps: %8d, total episode: %4d, value_loss: %f, policy_loss: %f, n_updates: %f\r" % (
            sampling_step, total_episode, statistics[2][1], statistics[3][1], statistics[4][1]), end="", flush=True)

        if sampling_step % args.evaluation_interval == 0:
            score = validate_step(env, agent, 100)
            print("\nEpisode: %5d, R(mean/std): %s" %
                  (total_episode, score))
            
            if not best_score or score[0] >= best_score[0]:
                filename = "agent_best.dat"
                agent.save(filename)
                print("new best score")
        


if __name__ == "__main__":
    args = parse_args("Train RL models using PFRL under Gymnasium")

    pprint_summary("PFRL Model Training", **args.__dict__)
    main(args)
