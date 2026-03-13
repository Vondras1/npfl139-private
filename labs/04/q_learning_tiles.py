#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import os

import npfl139
npfl139.require_version("2526.4")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.05, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.7, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=3500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--tiles", default=12, type=int, help="Number of tiles.")

import os
import re
import json
import numpy as np
import argparse


class AgentSaver:
    def __init__(self, args, main_folder):
        self.recodex = args.recodex
        self.main_folder = main_folder

    def save_next_agent(self, W: np.ndarray, args: argparse.Namespace) -> None:
        """
        Find the highest numbered agent folder in base_folder and save the new agent as +1.
        """

        os.makedirs(self.main_folder, exist_ok=True)

        pattern = re.compile(r"agent_(\d+)")
        max_id = 0

        for name in os.listdir(self.main_folder):
            match = pattern.fullmatch(name)
            if match:
                agent_id = int(match.group(1))
                max_id = max(max_id, agent_id)

        new_id = max_id + 1
        new_folder = os.path.join(self.main_folder, f"agent_{new_id}")

        AgentSaver.save_agent(new_folder, W, args)

        print(f"Agent saved to {new_folder}")

    def save_agent(self, folder_path: str, W: np.ndarray, args: argparse.Namespace) -> None:
        """Save W, and parser arguments into one folder."""

        os.makedirs(folder_path, exist_ok=True)

        w_path = os.path.join(folder_path, "W.npy")
        args_path = os.path.join(folder_path, "args.json")

        np.save(w_path, W)

        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)

    def load_agent(self, agent_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load W, and if not running in racodex also saved parser arguments from one folder."""

        if self.recodex:
            w_path = "W.npy"
            W = np.load(w_path)
            return W, None
        
        w_path = os.path.join(self.main_folder, agent_name, "W.npy")
        args_path = os.path.join(self.main_folder, agent_name, "args.json")

        W = np.load(w_path)
        with open(args_path, "r", encoding="utf-8") as f:
            saved_args = json.load(f)

        return W, saved_args


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.observation_space.nvec[-1], env.action_space.n])
    epsilon = args.epsilon

    if args.recodex:
        training = False
    else:
        training = True

    while training:
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            # TODO: Choose an epsilon-greedy action.
            q_vals = np.sum(W[state, :], axis=0)
            if np.random.rand() < epsilon:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = np.argmax(q_vals)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TODO: Update the action-value estimates
            max_q_val_next = 0 if done else np.max(np.sum(W[next_state], axis=0))
            W[state, action] += (args.alpha / len(state)) * (reward + args.gamma * max_q_val_next - q_vals[action])

            state = next_state

        # evaluate and quit training if target reached
        if env.episode % 500 == 0:
            returns = []
            for _ in range(100):
                state, done = env.reset()[0], False
                episode_return = 0
                while not done:
                    # TODO: Choose a greedy action
                    action = np.argmax(np.sum(W[state, :], axis=0))
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_return += reward
                
                returns.append(episode_return)

            mean_return = np.mean(returns)
            print("Evaluation return:", mean_return)
            if mean_return > -105:
                print("Target reached, stopping training.")
                break

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
    
    agent = AgentSaver(args=args, main_folder="04/q_tiles_agents")
    if training:
        # Save final model
        agent.save_next_agent(W, args)
    else:
        W, _ = agent.load_agent(agent_name="agent_1")

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            action = np.argmax(np.sum(W[state, :], axis=0))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("npfl139/MountainCar1000-v0"), tiles=main_args.tiles),
        main_args.seed, main_args.render_each,
    )

    main(main_env, main_args)
