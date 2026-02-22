#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=5000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.15, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=1, type=float, help="Discount factor")



def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # TODO:
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    Q = np.zeros(shape=(n_states, n_actions))
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.
    C = np.zeros(shape=(n_states, n_actions))
    # - Create returns to save G
    returns = np.zeros(shape=(n_states, n_actions))

    for _ in range(args.episodes):
        # TODO: Perform an episode, collecting states, actions and rewards.

        actions = []
        states = []
        rewards = []

        state, done = env.reset()[0], False
        while not done:
            # TODO: Compute `action` using epsilon-greedy policy.
            if args.epsilon < np.random.rand():
                # greedy
                action = np.argmax(Q[state, :])
            else:
                action = np.random.randint(low=0, high=n_actions)

            # Perform the action.
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Keep track of rewards, actions and states
            rewards.append(reward)
            actions.append(action)
            states.append(state)

            state = next_state

        # TODO: Compute returns from the received rewards and update Q and C. Backtrack
        G = 0
        for i in range(len(states)-1, -1, -1):
            C[states[i], actions[i]] += 1

            G = args.gamma * G + rewards[i]
            returns[states[i], actions[i]] += G

            Q[states[i], actions[i]] = returns[states[i], actions[i]]/C[states[i], actions[i]]


    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            action = np.argmax(Q[state, :])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), main_args.seed, main_args.render_each)

    main(main_env, main_args)
