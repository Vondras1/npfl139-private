#!/usr/bin/env python3
# Team:
# 1ac5d633-f96f-42a3-846d-31bcb01d041f
# e0cfa255-0259-11eb-9574-ea7484399335
# 9fafb47f-e1c5-4d7c-8ce5-8a6f5bdcd751

import argparse
import collections
import copy
import json

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="HalfCheetah-v5", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")

parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate after this many vectorized steps.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Number of evaluation episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--replay_buffer_size", default=300_000, type=int, help="Replay buffer size.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--model_path", default="cheetah_ddpg_actor.pt", type=str, help="Path to the actor model.")
parser.add_argument("--envs", default=16, type=int, help="Number of parallel environments.")
parser.add_argument("--exploration_noise", default=0.1, type=float, help="Stddev of exploration Gaussian noise.")
parser.add_argument("--min_buffer_size", default=20000, type=int, help="Minimum replay buffer size before training.")
parser.add_argument("--target_return", default=8200, type=float, help="Optional target return for stopping.")


class Agent:
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self._gamma = args.gamma
        self._target_tau = args.target_tau
        self._exploration_noise = args.exploration_noise

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        action_high = torch.tensor(env.action_space.high, dtype=torch.float32)
        action_scale = (action_high - action_low) / 2
        action_offset = (action_high + action_low) / 2

        self._action_low = action_low.to(self.device)
        self._action_high = action_high.to(self.device)

        class Actor(torch.nn.Module):
            def __init__(self, hidden_layer_size: int):
                super().__init__()
                self._network = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, hidden_layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_layer_size, hidden_layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_layer_size, action_dim),
                    torch.nn.Tanh(),
                )
                self.register_buffer("action_scale", action_scale)
                self.register_buffer("action_offset", action_offset)

            def forward(self, states: torch.Tensor) -> torch.Tensor:
                return self.action_scale * self._network(states) + self.action_offset

        class Critic(torch.nn.Module):
            def __init__(self, hidden_layer_size: int):
                super().__init__()
                self._network = torch.nn.Sequential(
                    torch.nn.Linear(state_dim + action_dim, hidden_layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_layer_size, hidden_layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_layer_size, 1),
                )

            def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
                return self._network(torch.cat([states, actions], dim=-1)).squeeze(-1)

        self._actor = Actor(args.hidden_layer_size).to(self.device)
        self._target_actor = copy.deepcopy(self._actor)

        self._critic = Critic(args.hidden_layer_size).to(self.device)
        self._target_critic = copy.deepcopy(self._critic)

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=args.learning_rate)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=args.learning_rate)

        self._mse = torch.nn.MSELoss()

    def save_actor(self, path: str) -> None:
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str) -> None:
        self._actor.load_state_dict(torch.load(path, map_location=self.device))

    @staticmethod
    def save_args(path: str, args: argparse.Namespace) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(vars(args), file, ensure_ascii=False, indent=2)

    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # Critic update
        predicted_returns = self._critic(states, actions)
        critic_loss = self._mse(predicted_returns, returns)

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        # Actor update
        predicted_actions = self._actor(states)
        actor_loss = -self._critic(states, predicted_actions).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        # EMA target update
        npfl139.update_params_by_ema(self._target_actor, self._actor, self._target_tau)
        npfl139.update_params_by_ema(self._target_critic, self._critic, self._target_tau)

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_mean_actions(self, states: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self._actor(states)

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_sampled_actions(self, states: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            actions = self._actor(states)
            noise = torch.randn_like(actions) * self._exploration_noise
            actions = torch.clamp(actions + noise, self._action_low, self._action_high)
            return actions

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_target_values(self, states: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            next_actions = self._target_actor(states)
            return self._target_critic(states, next_actions)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    agent = Agent(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0.0, False
        while not done:
            # Predict an action using the greedy policy.
            action = agent.predict_mean_actions(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Evaluation in ReCodEx
    if args.recodex:
        # Load a pretrained model and perform evaluation.
        agent.load_actor(args.model_path)
        while True:
            evaluate_episode(start_evaluation=True)

    # Training
    vector_env = gym.make_vec(
        args.env,
        args.envs,
        gym.VectorizeMode.ASYNC,
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )

    replay_buffer = npfl139.ReplayBuffer(args.replay_buffer_size, args.seed)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    state = vector_env.reset(seed=args.seed)[0]
    best_mean_return = -np.inf

    while True:
        # Collect experience and train
        for _ in range(args.evaluate_each):
            action = agent.predict_sampled_actions(state)

            next_state, reward, terminated, truncated, _ = vector_env.step(action)
            done = terminated | truncated

            replay_buffer.append_batch(Transition(state, action, reward, done, next_state))
            state = next_state

            if len(replay_buffer) >= max(args.min_buffer_size, args.batch_size):
                states, actions, rewards, dones, next_states = replay_buffer.sample(args.batch_size)

                next_values = agent.predict_target_values(next_states)
                returns = rewards + args.gamma * next_values * (1 - dones.astype(np.float32))

                agent.train(states, actions, returns)

        # Evaluate
        evaluation_returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        mean_return = np.mean(evaluation_returns)
        print(f"Evaluation after training block: {mean_return:.2f}")

        if mean_return > best_mean_return:
            best_mean_return = mean_return
            agent.save_actor(args.model_path)
            agent.save_args(args.model_path + ".json", args)
            print(f"-> New best mean return {best_mean_return:.2f}, model saved.")

        if args.target_return is not None and mean_return >= args.target_return:
            break

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)
