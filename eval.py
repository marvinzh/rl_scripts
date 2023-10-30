import argparse
import time
import gymnasium as gym
import numpy as np
from utils.general import pprint_summary
from commons import load_agent, load_env


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("agent", choices=["PPO"])
    parser.add_argument(
        "env", type=str, help="Environment name of OpenAI Gymnasium")
    parser.add_argument("profile", help="profile for consutrcting an agent")
    parser.add_argument("--model", "-m", type=str, help="model file of PFRL")
    parser.add_argument("--fps", default=20, type=int, choices=range(1, 61),
                        help="FPS(1 ~ 60) to control the speed of simulation")
    parser.add_argument("--turns", default=1, type=int, help="turns for game")

    parser.add_argument("--verbose", "-v", default=False,
                        action="store_true", help="display the step log")
    parser.add_argument("--visual", "-vv", default=False, action="store_true",
                        help="display the step log and visualize the simulation")

    return parser.parse_args()


def main(args):
    env = load_env(args.env, args.visual)
    agent = load_agent(args.profile)
    if args.model:
        agent.load(args.model)

    if args.turns == 0:
        args.turns = float("inf")

    i_turn = 0
    Rs = []
    while i_turn < args.turns:
        i_turn += 1
        R = play_a_round(env, agent, {
            "i_turn": i_turn,
        }, args.fps)
        Rs.append(R)

    env.close()
    print("\nRs(mean/std): %s" % ([np.mean(Rs), np.std(Rs)]))


def play_a_round(env: gym.Env, agent, info: dict, fps=24):
    obs, _ = env.reset()
    R = 0
    for i_step in range(env.spec.max_episode_steps):
        with agent.eval_mode():
            action = agent.act(obs)
            obs, r, done, truncated, _ = env.step(action)
            R += r
            print("#Round: %3d, step: %4d, Action: %2d, r: %2d, R: %3d\r" %
                  (info["i_turn"], i_step, action, r, R), end="", flush=True)
            if fps > 0:
                time.sleep(1/fps)
            if done or truncated:
                break

    return R


if __name__ == "__main__":
    args = parse_args("Evaluate RL models trained by PFRL under Gymnasium")

    pprint_summary("PFRL Model Evaluation", **args.__dict__)
    main(args)
