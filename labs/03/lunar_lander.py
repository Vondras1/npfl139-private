#!/usr/bin/env python3
# Team:
# 1ac5d633-f96f-42a3-846d-31bcb01d041f
# 9fafb47f-e1c5-4d7c-8ce5-8a6f5bdcd751

import argparse
import re
from time import time

import os
import json

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2526.3")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.15, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--episodes", default=50000, type=int, help="Training episodes.")
parser.add_argument("--warmup_episodes", default=0, type=int, help="Warmup episodes.")
parser.add_argument("--mode", default="qlearn", type=str, help="Mode (qlearn/tree_backup/monte_carlo).")
parser.add_argument("--load_agent", default=None, type=str, help="Load pretrained agent 'agent_#'")

# --load_agent agent_1 --mode qlearn --warmup_episodes 0 --episodes 50000
# python 03/lunar_lander.py --load_agent agent_4 --warmup_episodes 0 --episodes 200000 --mode monte_carlo

def save_next_agent(base_folder: str, Q: np.ndarray, C: np.ndarray, args: argparse.Namespace) -> None:
    """
    Find the highest numbered agent folder in base_folder and save the new agent as +1.
    """

    os.makedirs(base_folder, exist_ok=True)

    pattern = re.compile(r"agent_(\d+)")
    max_id = 0

    for name in os.listdir(base_folder):
        match = pattern.fullmatch(name)
        if match:
            agent_id = int(match.group(1))
            max_id = max(max_id, agent_id)

    new_id = max_id + 1
    new_folder = os.path.join(base_folder, f"agent_{new_id}")

    save_agent(new_folder, Q, C, args)

    print(f"Agent saved to {new_folder}")

def save_agent(folder_path: str, Q: np.ndarray, C: np.ndarray, args: argparse.Namespace) -> None:
    """Save Q, C, and parser arguments into one folder."""
    os.makedirs(folder_path, exist_ok=True)

    q_path = os.path.join(folder_path, "Q.npy")
    c_path = os.path.join(folder_path, "C.npy")
    args_path = os.path.join(folder_path, "args.json")

    np.save(q_path, Q)
    np.save(c_path, C)

    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)


def load_agent(folder_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load Q, C, and saved parser arguments from one folder."""
    q_path = os.path.join(folder_path, "Q.npy")
    c_path = os.path.join(folder_path, "C.npy")
    args_path = os.path.join(folder_path, "args.json")

    Q = np.load(q_path)
    C = np.load(c_path)

    with open(args_path, "r", encoding="utf-8") as f:
        saved_args = json.load(f)

    return Q, C, saved_args

def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)

def generate_expert_episode(env: npfl139.EvaluationEnv) -> list[tuple[np.ndarray, int, float]]:
        episode = env.expert_episode() # (state, action, reward) tuples

        # Trajectory buffers:
        # S[t] = state at time t
        # A[t] = action taken at time t
        # R[t+1] = reward observed after action A[t]
        S = []
        A = []
        R = [0] # FIXME

        for s, a, r in episode:
            S.append(s)
            if a is not None:
                A.append(a)
                R.append(float(r))

        return S, A, R

def generate_e_greedy(env: npfl139.EvaluationEnv, Q: np.ndarray, args: argparse.Namespace) -> list[tuple[np.ndarray, int, float]]:
    state, done = env.reset()[0], False

    S = []
    A = []
    R = [0]

    while not done:
        S.append(state)

        if np.random.rand() < args.epsilon:
            action = env.action_space.sample()
        else:
            action = argmax_with_tolerance(Q[state])

        A.append(action)

        state, reward, terminated, truncated, _ = env.step(action)
        R.append(float(reward))

        done = terminated or truncated

    return S, A, R


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # TODO: Implement a suitable RL algorithm and train the agent.
    if args.load_agent:
        Q, C, _ = load_agent(f"03/lunar_lander_agent/{args.load_agent}")
    else:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        C = np.zeros((env.observation_space.n, env.action_space.n))

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent

        # Final evaluation
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # TODO: Choose a greedy action
                action = argmax_with_tolerance(Q[state])
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # The target policy is either the behavior policy (if not `args.off_policy`),
    # or the greedy policy (if `args.off_policy`).
    def compute_target_policy(Q: np.ndarray) -> np.ndarray:
        target_policy = np.eye(env.action_space.n)[argmax_with_tolerance(Q, axis=-1)]
        target_policy = (1 - args.epsilon) * target_policy + args.epsilon / env.action_space.n
        return target_policy

    training = True
    start_time = time()
    episode_count = 0
    while training and episode_count < args.episodes:
        warmingup = (episode_count < args.warmup_episodes)

        # Generate an episode using the behavior policy
        if warmingup:
            # To generate an expert episode, you can use the following:
            S, A, R = generate_expert_episode(env)
        else:
            S, A, R = generate_e_greedy(env, Q, args)
                    
        T = len(A)

        if warmingup:
            G = 0
            for t in reversed(range(T)):
                G = R[t + 1] + args.gamma * G
                # if C[S[t], A[t]] < 800:
                C[S[t], A[t]] += 1
                Q[S[t], A[t]] += (1 / C[S[t], A[t]]) * (G - Q[S[t], A[t]])

        elif not warmingup and args.mode == "tree_backup":
            for tau in range(T): 
                end = min(tau + args.n, T)

                if tau + args.n >= T:
                    G = R[T]

                else:
                    expected_value = 0
                    target_policy = compute_target_policy(Q)
                    for a in range(env.action_space.n):
                        expected_value += target_policy[S[end], a] * Q[S[end], a]
                    G = R[end] + args.gamma * expected_value
                
                for i in range(end-1, tau, -1):
                    expected_value = 0
                    target_policy = compute_target_policy(Q)
                    for a in range(env.action_space.n):
                        if a == A[i]: # Action taken in the episode
                            expected_value += target_policy[S[i], a] * G
                        else:
                            expected_value += target_policy[S[i], a] * Q[S[i], a]
                    G = R[i+1] + args.gamma * expected_value
                
                Q[S[tau], A[tau]] += args.alpha * (G - Q[S[tau], A[tau]])
        
        elif not warmingup and args.mode == "qlearn":
            for t in range(T):
                if t == T - 1:
                    target = R[t + 1]
                else:
                    target = R[t + 1] + args.gamma * np.max(Q[S[t + 1]])
                Q[S[t], A[t]] += args.alpha * (target - Q[S[t], A[t]])

        elif not warmingup and args.mode == "monte_carlo":
            G = 0
            for t in reversed(range(T)):
                G = R[t + 1] + args.gamma * G
                C[S[t], A[t]] += 1
                Q[S[t], A[t]] += (1 / C[S[t], A[t]]) * (G - Q[S[t], A[t]]) # TODO consider replacing (1 / C[S[t], A[t]]) with args.alpha

        episode_count += 1

    return Q, C


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteLunarLanderWrapper(gym.make("LunarLander-v3")), main_args.seed, main_args.render_each, report_each=10, evaluate_for=2000)

    Q, C = main(main_env, main_args)

    # Save the agent
    if not main_args.recodex:
        save_next_agent("03/lunar_lander_agent", Q, C, main_args)
