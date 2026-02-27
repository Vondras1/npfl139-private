#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2526.2")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.4, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.8, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_min", default=0.05, type=float, help="Minimal exploration factor.")
parser.add_argument("--gamma", default=0.9, type=float, help="Discounting factor.")
parser.add_argument("--episodes", default=10000, type=int, help="Episodes to perform.")


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # TODO: Variable creation and initialization
    Q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64) # Initialize Q(s,a)
    N = np.zeros_like(Q_table, dtype=np.int32) # Initialize N(s,a)

    training = True
    episode = 0
    while training:
        # Perform episode
        epsilon = max(args.epsilon * (0.99985 ** episode), args.epsilon_min)
        # print(f"Episode {episode}, epsilon: {epsilon:.4f}")
        state, done = env.reset()[0], False
        while not done:
            # TODO: Perform an action.
            if np.random.uniform() < epsilon:
                # Explore: choose a random action
                action = np.random.randint(env.action_space.n)
            else:
                action = np.argmax(Q_table[state])

            N[state, action] += 1
            alpha = 1.0 / np.sqrt(N[state, action])
            # alpha = min(args.alpha, alpha) # Optional: cap the learning rate to a maximum value
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TODO: Update the action-value estimates
            Q_table[state, action] = Q_table[state, action] + alpha * (reward + args.gamma * np.max(Q_table[next_state]) - Q_table[state, action])

            state = next_state
        
        episode += 1
        if episode >= args.episodes:
            training = False

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            action = np.argmax(Q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("npfl139/MountainCar1000-v0")),
        main_args.seed, main_args.render_each,
    )

    main(main_env, main_args)
