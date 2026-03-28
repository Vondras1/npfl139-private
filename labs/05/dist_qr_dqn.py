#!/usr/bin/env python3
import argparse
import collections
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.5")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=False, action="store_true", help="Verify the loss computation")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=800, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--kappa", default=1, type=float, help="The quantile Huber loss threshold.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate.")
parser.add_argument("--quantiles", default=100, type=int, help="Number of quantiles.")
parser.add_argument("--target_update_freq", default=..., type=int, help="Target update frequency.")


class Network:
    device = torch.device("cpu")
    # Use the following line instead to use GPU if available.
    # device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model and store it as `self._model`. The model
        # should compute `args.quantiles` quantiles for each action, so for input
        # of shape `[batch_size, *env.observation_space.shape]`, the output should
        # have the shape `[batch_size, env.action_space.n, args.quantiles]`.
        # The module `torch.nn.Unflatten` might come handy.
        self._model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n*args.quantiles),
            torch.nn.Unflatten(1, (int(env.action_space.n), int(args.quantiles)))
        )
        self._model.to(self.device)

        # Store the discount factor and the quantile Huber loss threshold.
        self.gamma = args.gamma
        self.kappa = args.kappa

        # TODO(q_network): Define a suitable optimizer from `torch.optim`.
        self._optimizer = torch.optim.AdamW(lr=args.learning_rate, params=self._model.parameters())

    @staticmethod
    def compute_loss(
        states_quantiles: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
        next_states_quantiles: torch.Tensor, gamma: float, kappa: float,
    ) -> torch.Tensor:
        # TODO: Implement the loss computation according to the QR-DQN-kappa algorithm.
        # - The `states_quantiles` are current state quantiles, of shape `[batch_size, actions, quantiles]`.
        # - The `actions` are the integral actions taken in the states, of shape `[batch_size]`.
        # - The `rewards` are the rewards obtained after taking the actions, of shape `[batch_size]`.
        # - The `dones` are `torch.float32` indicating whether the episode ended, of shape `[batch_size]`.
        # - The `next_states_quantiles` are next states quantiles, of shape `[batch_size, actions, quantiles]`.
        #   Because they should not be backpropagated through, use an appropriate `.detach()` call.
        # - The non-negative `kappa` is the threshold for the quantile Huber loss (delta in PyTorch terminology).
        #   When `kappa=0` is passed, the standard (non-Huber) quantile regression loss should be used.
        # The number of quantiles is given by the shape of `states_quantiles`, and the quantiles
        # tau_1, ..., tau_N are uniformly spaced between 0 (exclusive) and 1 (inclusive), so tau_i = i / N.
        # The resulting loss should be the mean over all trained quantiles and all batch examples,
        # unlike the algorithm in the paper, which computes a sum over the trained quantiles.
        
        batch_size = actions.shape[0]
        batch_indices = torch.arange(batch_size, device=actions.device)
        
        # ----- Choose one quantile ditribution for all actions taken. => Every batch sample has one distribution # [batch, # of quantiles]
        current_quantiles = states_quantiles[batch_indices, actions]

        # ----- Target quantiles -----
        next_states_quantiles = next_states_quantiles.detach()
        # Compute Q values from each ditribution (One value for every action)
        Q_next = next_states_quantiles.mean(dim=-1)
        # Take the greedy action
        greedy_actions = torch.argmax(Q_next, dim=1)
        # Choose the appropriate quantile ditribution for greedy actions
        greedy_actions_distribution = next_states_quantiles[batch_indices, greedy_actions]

        # Perform Bellman update
        target_quantiles = rewards[:, None] + gamma * (1 - dones[:, None]) * greedy_actions_distribution
        
        # ----- Compute quantile-huber loss -----
        # pairwise differences: [B, N, N]
        diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)

        # Taus: [N]
        num_quantiles = current_quantiles.shape[1]
        # taus = torch.arange(1, num_quantiles + 1, device=actions.device, dtype=current_quantiles.dtype) / num_quantiles
        # taus = taus.view(1, num_quantiles, 1)  # [1, N, 1]
        taus = ((torch.arange(num_quantiles, device=actions.device, dtype=current_quantiles.dtype) + 0.5)/ num_quantiles).view(1, num_quantiles, 1)  # [1, N, 1]

        # Quantile weights |tau_i - 1[u_ij < 0]|: [B, N, N]
        weight = torch.abs(taus - (diff.detach() < 0).to(current_quantiles.dtype))

        # Base loss
        if kappa == 0:
            base_loss = diff.abs()
        else:
            abs_diff = diff.abs()
            base_loss = torch.where(
                abs_diff <= kappa,
                0.5 * diff.pow(2),
                kappa * (abs_diff - 0.5 * kappa),
            )

        # Final quantile loss: mean over target quantiles, mean over trained quantiles, mean over batch
        loss = (weight * base_loss).mean(dim=2).mean(dim=1).mean()

        return loss

    # The training function defers the computation to the `compute_loss` method.
    #
    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
              dones: torch.Tensor, next_states: torch.Tensor) -> None:
        self._model.train()
        # Pass all arguments to the `compute_loss` method.
        loss = self.compute_loss(
            self._model(states), actions, rewards, dones, self._model(next_states), self.gamma, self.kappa)
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            # TODO: Return all predicted Q-values for the given states.
            quantiles = self._model(states)
            Q_values = quantiles.mean(dim=-1)
            return Q_values

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> Callable | None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Create evaluation env
    eval_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    # When the `args.verify` is set, just return the loss computation function for validation.
    if args.verify:
        return Network.compute_loss

    # Construct the network
    network = Network(env, args)

    # # Construct the target network
    # target_network = Network(env, args)
    # target_network.copy_weights_from(network)

    # Replay memory; the `max_length` parameter is its maximum capacity.
    replay_buffer = npfl139.ReplayBuffer(max_length=1_000_000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    training = True
    while training:
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            # TODO(q_network): Choose an action.
            # You can compute the q_values of a given state by
            #   q_values = network.predict(state[np.newaxis])[0]
            if np.random.rand() < epsilon:
                action = np.random.randint(0, env.action_space.n)
            else:
                q_values = network.predict(state[np.newaxis])[0]
                action = np.argmax(q_values)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # TODO: If the `replay_buffer` is large enough, perform training by
            # sampling a batch of `args.batch_size` uniformly randomly chosen transitions
            # and calling `network.train(states, actions, rewards, dones, next_states)`.
            #
            # The `replay_buffer` offers a method with signature
            #   sample(self, size, replace=True) -> NamedTuple
            # which returns uniformly selected batch of `size` transitions, either with
            # replacement (which is faster, and hence the default) or without.
            # The returned batch is a `Transition` named tuple, each field being
            # a NumPy array containing a batch of corresponding transition components.
            if len(replay_buffer) > args.batch_size*5:
                # TRAIN
                batch = replay_buffer.sample(args.batch_size, replace=True)
                # network.train(samples)
                network.train(
                    batch.state, batch.action, batch.reward, batch.done, batch.next_state
                )

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        # evaluate and quit training if target reached
        if env.episode % 200 == 0:
            returns = []
            for _ in range(100):
                state, done = eval_env.reset()[0], False
                episode_return = 0
                while not done:
                    # TODO: Choose a greedy action
                    q_values = network.predict(state[np.newaxis])[0]
                    action = np.argmax(q_values)  
                    state, reward, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated
                    episode_return += reward
                
                returns.append(episode_return)

            mean_return = np.mean(returns)
            print("Evaluation return:", mean_return)
            if mean_return > 450:
                torch.save(network._model.state_dict(), "dist_qr_dqn.pt")
                print("Target reached, stopping training.")
                break

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO(q_network): Choose (greedy) action
            q_values = network.predict(state[np.newaxis])[0]
            action = np.argmax(q_values) 
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    result = main(main_env, main_args)
    if main_args.verify:
        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-1.4, 0.1, 0.8], [-1.2, 0.1, 1.1]]]),
            actions=torch.tensor([1]), rewards=torch.tensor([-1.5]), dones=torch.tensor([0.]),
            next_states_quantiles=torch.tensor([[[-0.4, 0.1, 0.4], [-0.5, 1.0, 1.6]]]),
            gamma=0.2, kappa=1.5).numpy(force=True), 0.3294963, atol=1e-5)

        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-0.0, 0.1, 1.2], [-1.8, -0.2, -0.1]],
                                           [[-0.3, 0.5, 1.3], [-1.4, -0.7, -0.1]],
                                           [[-0.3, -0.0, 1.9], [-1.1, -0.2, -0.1]]]),
            actions=torch.tensor([1, 0, 1]), rewards=torch.tensor([0.5, 1.4, 0.1]), dones=torch.tensor([0., 0., 1.]),
            next_states_quantiles=torch.tensor([[[-1.1, 0.2, 0.3], [-0.4, 1.1, 1.3]],
                                                [[-0.6, -0.5, 2.0], [-0.3, 0.2, 0.4]],
                                                [[-0.9, 0.7, 2.3], [-0.3, 0.7, 0.7]]]),
            gamma=0.8, kappa=0.0).numpy(force=True), 0.4392593, atol=1e-5)

        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-0.8, -0.5, -0.0, 0.3], [-0.7, -0.2, -0.2, 1.6]],
                                           [[-1.5, -1.4, -0.6, 0.1], [-2.1, -1.5, -0.3, 0.3]]]),
            actions=torch.tensor([1, 0]), rewards=torch.tensor([-0.0, 0.7]), dones=torch.tensor([1., 0.]),
            next_states_quantiles=torch.tensor([[[-1.2, 0.3, 0.4, 0.7], [-1.2, -0.1, 0.4, 2.2]],
                                                [[-1.5, 0.2, 0.2, 0.5], [-0.9, 0.4, 0.5, 1.3]]]),
            gamma=0.3, kappa=3.5).numpy(force=True), 0.2906375, atol=1e-5)
