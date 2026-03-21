#!/usr/bin/env python3
import argparse
import collections

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.4")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=6000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.98, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=4000, type=int, help="Target update frequency.")


class Network:
    device = torch.device("cpu")
    # Use the following line instead to use GPU if available.
    # device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model and store it as `self._model`.
        self._model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n)
        ).to(self.device)

        # TODO: Define a suitable optimizer from `torch.optim`.
        self._optimizer = torch.optim.AdamW(self._model.parameters(), args.learning_rate, weight_decay=1e-4)

        # TODO: Define the loss (most likely some `torch.nn.*Loss`).
        self._loss = torch.nn.MSELoss()

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, and include the index of the action to which
    #   the new q_value belongs
    # The code below implements the first option, but you can change it if you want.
    #
    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, q_values: torch.Tensor) -> None:
        self._model.train()
        predictions = self._model(states)
        loss = self._loss(predictions, q_values)
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            return self._model(states)

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace, eval_env) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the network
    network = Network(env, args)

    # Construct the target network
    target_network = Network(env, args)
    target_network.copy_weights_from(network)

    # Replay memory; the `max_length` parameter is its maximum capacity.
    replay_buffer = npfl139.ReplayBuffer(max_length=1_000_000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    training = True

    if args.recodex:
        training = False
    else:
        training = True

    train_steps = 0
    while training:
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            # TODO: Choose an action.
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

            # TODO: If the `replay_buffer` is large enough, perform training using
            # a batch of `args.batch_size` uniformly randomly chosen transitions.
            #
            # The `replay_buffer` offers a method with signature
            #   sample(self, size, replace=True) -> NamedTuple
            # which returns uniformly selected batch of `size` transitions, either with
            # replacement (which is faster, and hence the default) or without.
            # The returned batch is a `Transition` named tuple, each field being
            # a NumPy array containing a batch of corresponding transition components.

            # After you compute suitable targets, you can train the network by
            #   network.train(...)
            if len(replay_buffer) > args.batch_size*5:
                # TRAIN
                samples = replay_buffer.sample(args.batch_size, replace=True)

                q_values = network.predict(samples.state)
                q_values_next = target_network.predict(samples.next_state)

                targets = samples.reward + args.gamma * np.max(q_values_next, axis=1) * (~samples.done)

                q_values[np.arange(args.batch_size), samples.action] = targets

                network.train(samples.state, q_values)

                train_steps += 1
                if train_steps % args.target_update_freq == 0:
                    print("Update")
                    target_network.copy_weights_from(network)

            state = next_state

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
                torch.save(network._model.state_dict(), "q_network_cartpole.pt")
                print("Target reached, stopping training.")
                break

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

    if not training:
        network._model.load_state_dict(torch.load("04/q_network_cartpole.pt", map_location=Network.device))
        network._model.eval()

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            q_values = network.predict(state[np.newaxis])[0]
            action = np.argmax(q_values)            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)
    eval_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)
    
    main(main_env, main_args, eval_env)
