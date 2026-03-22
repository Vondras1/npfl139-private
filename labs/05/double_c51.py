#!/usr/bin/env python3
# Time to solve the task: 4h
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
parser.add_argument("--atoms", default=50, type=int, help="Number of atoms.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=500, type=int, help="Target update frequency.")


class Network:
    device = torch.device("cpu")
    # Use the following line instead to use GPU if available.
    # device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model and store it as `self._model`. The model
        # should compute `args.atoms` logits for each action, so for input of shape
        # `[batch_size, *env.observation_space.shape]`, the output should have
        # the shape `[batch_size, env.action_space.n, args.atoms]`. The module
        # `torch.nn.Unflatten` might come handy.
        self._model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n*args.atoms),
            torch.nn.Unflatten(1, (int(env.action_space.n), int(args.atoms)))
        )

        # Create `self._model.atoms` as uniform grid from 0 to 500 with `args.atoms` elements.
        # We create them as a buffer in `self._model` so they are automatically moved with `.to`.
        self._model.register_buffer("atoms", torch.linspace(0, 500, args.atoms))

        self._model.to(self.device)

        # Store the discount factor.
        self.gamma = args.gamma

        # TODO(q_network): Define a suitable optimizer from `torch.optim`.
        self._optimizer = torch.optim.AdamW(lr=args.learning_rate, params=self._model.parameters())

    @staticmethod
    def compute_loss(
        states_logits: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
        next_states_logits_online: torch.Tensor, next_states_logits_target: torch.Tensor, atoms: torch.Tensor, gamma: float,
    ) -> torch.Tensor:
        # TODO: Implement the loss computation according to the C51 algorithm.
        # - The `states_logits` are current state logits of shape `[batch_size, actions, atoms]`.
        # - The `actions` are the integral actions taken in the states, of shape `[batch_size]`.
        # - The `rewards` are the rewards obtained after taking the actions, of shape `[batch_size]`.
        # - The `dones` are `torch.float32` indicating whether the episode ended, of shape `[batch_size]`.
        # - The `next_states_logits_online` are logits of the next states, of shape `[batch_size, actions, atoms]`.
        #   Because they should not be backpropagated through, use an appropriate `.detach()` call.
        # - The `next_states_logits_target` are logits of the target next states, of shape `[batch_size, actions, atoms]`.
        #   (obtained by target network), use an appropriate `.detach()` call.
        # - The `atoms` are the atom values. Your implementation must handle any number of atoms. The
        #   `atoms[0]` is V_MIN (the minimum atom value), `atoms[-1]` is V_MAX (the maximum atom value),
        #   and use `atoms[1] - atoms[0]` as the distance between two consecutive atoms. You can
        #   assume that one of the atoms is always 0.
        # The resulting loss should be the mean of the cross-entropy losses of the individual batch examples.
        #
        # Your implementation most likely needs to be vectorized to pass ReCodEx time limits. Note that you
        # can add given values to a vector of (possibly repeating) tensor indices using `scatter_add_`.
        
        batch_size = actions.shape[0]
        batch_indices = torch.arange(batch_size, device=actions.device)

        # [batch, # of atoms], for every sample one distribution
        train_logits = states_logits[batch_indices, actions]

        V_min = atoms[0]
        V_max = atoms[-1]
        delta_z = atoms[1] - atoms[0]

        # Get probabilities from next state logits
        next_probs_online = torch.softmax(next_states_logits_online.detach(), dim=-1)   # [B, A, N]
        next_probs_target = torch.softmax(next_states_logits_target.detach(), dim=-1)   # [B, A, N]
        # Compute Q values from each ditribution (One value for every action). From ONLINE network
        Qs_next = (next_probs_online*atoms).sum(dim=-1)
        # Choose the best action
        greedy_actions = torch.argmax(Qs_next, dim=1)
        # From target network take the distribution of the greedy action in the next state
        greedy_actions_distribution = next_probs_target[batch_indices, greedy_actions]

        # shift the next-state atom support by reward and discount
        tz = rewards[:, None] + gamma * (1 - dones[:, None]) * atoms[None, :]
        tz = torch.clip(tz, V_min, V_max)

        # project shifted atom values onto the fixed atom grid
        b = (tz - V_min)/delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        # Target distribution
        m = torch.zeros_like(greedy_actions_distribution)   # [B, N]

        # ditribute the probability # HAHAHAHAHAHAHAHA
        offset = (torch.arange(batch_size, device=actions.device) * atoms.shape[0])[:, None]
        offset = offset.expand_as(l)

        m.view(-1).scatter_add_(
            0,
            (l + offset).reshape(-1),
            (greedy_actions_distribution * (u.float() - b)).reshape(-1)
        )
        m.view(-1).scatter_add_(
            0,
            (u + offset).reshape(-1),
            (greedy_actions_distribution * (b - l.float())).reshape(-1)
        )
        eq_mask = (l == u)
        m.view(-1).scatter_add_(
            0,
            (l[eq_mask] + offset[eq_mask]).reshape(-1),
            greedy_actions_distribution[eq_mask].reshape(-1)
        )

        # Finall cross-entropy loss between the predicted probabilities `log_probs` and the target distribution `m`
        log_probs = torch.log_softmax(train_logits, dim=-1)
        loss = -(m * log_probs).sum(dim=1).mean()
        return loss


    # The training function defers the computation to the `compute_loss` method.
    def train(self, states, actions, rewards, dones, next_states, target_model) -> None:
        self._model.train()

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)

        states_logits = self._model(states)
        next_states_logits_online = self._model(next_states)

        with torch.no_grad():
            next_states_logits_target = target_model._model(next_states)

        loss = self.compute_loss(
            states_logits,
            actions,
            rewards,
            dones,
            next_states_logits_online,
            next_states_logits_target,
            self._model.atoms,
            self.gamma,
        )

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            # TODO: Return all predicted Q-values for the given states.
            logits = self._model(states)                      # [B, A, N]
            probs = torch.softmax(logits, dim=-1)             # [B, A, N]
            q_values = (probs * self._model.atoms).sum(dim=-1)  # [B, A]
            return q_values

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> Callable | None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    eval_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    # When the `args.verify` is set, just return the loss computation function for validation.
    if args.verify:
        return Network.compute_loss

    # Construct the network
    network = Network(env, args)

    # Construct the target network
    target_network = Network(env, args)
    target_network.copy_weights_from(network)

    # Replay memory; the `max_length` parameter is its maximum capacity.
    replay_buffer = npfl139.ReplayBuffer(max_length=1_000_000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    train_steps = 0
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
            # and calling `network.train(states, actions, rewards, dones, next_states, target_network)`.
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
                # Double C51:
                # - online network selects next action
                # - target network provides next-state distribution
                network.train(
                    batch.state,
                    batch.action,
                    batch.reward,
                    batch.done,
                    batch.next_state,
                    target_network,
                )

                train_steps += 1
                if train_steps % args.target_update_freq == 0:
                    target_network.copy_weights_from(network)
                    print("Weights updated.")

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
                torch.save(network._model.state_dict(), "double_c51.pt")
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
            states_logits=torch.tensor([[[-1.5, 1.2, -1.2], [-0.0, -1.8, -0.1]],
                                        [[-0.2, -0.3, 1.3], [0.5, -1.1, -0.7]],
                                        [[-0.1, 1.9, -0.0], [-0.3, -1.1, -0.1]]]),
            actions=torch.tensor([0, 1, 0]),
            rewards=torch.tensor([0.5, -0.2, 0.7]), dones=torch.tensor([1., 0., 0.]),
            next_states_logits=torch.tensor([[[1.1, 0.2, 0.3], [0.3, 1.1, 1.3]],
                                             [[-0.4, -0.5, -0.6], [2.0, 1.2, 0.4]],
                                             [[-0.3, -0.9, 2.3], [0.7, 0.7, -0.3]]]),
            atoms=torch.tensor([-2., -1., 0.]),
            gamma=0.3).numpy(force=True), 2.170941, atol=1e-5)

        np.testing.assert_allclose(result(
            states_logits=torch.tensor([[[0.1, 1.4, -0.5, -0.8], [0.3, -0.0, -0.2, -0.2]],
                                        [[1.2, -0.8, -1.4, -1.5], [0.1, -0.6, -2.1, -0.3]]]),
            actions=torch.tensor([0, 1]),
            rewards=torch.tensor([0.5, 0.6]), dones=torch.tensor([0., 0.]),
            next_states_logits=torch.tensor([[[0.8, 1.2, -1.2, 0.7], [0.3, 0.4, -1.2, 0.4]],
                                             [[-0.2, 1.0, -1.5, 0.2], [0.2, 0.5, 0.4, -0.9]]]),
            atoms=torch.tensor([-3., 0., 3., 6.]),
            gamma=0.2).numpy(force=True), 1.43398, atol=1e-5)
