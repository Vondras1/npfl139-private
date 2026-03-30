#!/usr/bin/env python3
import argparse
import collections
import os

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.6")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of value hidden layer.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=1000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=1000, type=int, help="Target update frequency.")
parser.add_argument("--max_episodes", default=5000, type=int, help="Maximum number of episodes.")
parser.add_argument("--evaluation_episodes_short", default=10, type=int, help="Number of evaluation episodes.")
parser.add_argument("--evaluation_episodes_long", default=100, type=int, help="Number of evaluation episodes.")
parser.add_argument("--pretrained_model_path", default="best_model.pt", type=str, help="Pre-trained model name.")
parser.add_argument("--load_model", default=True, type=bool, help="Load pre-trained model.")

class Network(torch.nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.args = args

        actions_n = env.action_space.n
        H, W, C = env.observation_space.shape # It should be [64x64x3]

        in_channels = C
        out_channels = 32
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=8, stride=4, padding=1, bias=False) # [16x16x32]
        self.relu = torch.nn.ReLU()

        in_channels = out_channels
        out_channels = 64
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False) # [8x8x64]

        in_channels = out_channels
        out_channels = 64
        self.conv3 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False) # [4x4x64]

        # in_channels = out_channels
        # out_channels = 64
        # self.conv4 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False) # [4x4x64]

        flattened_size = out_channels * (H // 16) * (W // 16)  # for 80x80 -> 64*5*5 = 1600 | for 64x64 -> 64*4*4 = 1024 | for 48x48 -> 64*3*3 = 576

        # Final linear layer
        self._final_linear = torch.nn.Sequential(
            torch.nn.Linear(flattened_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, actions_n),
        ).to(self.device)
        # self._final_linear = torch.nn.Linear(flattened_size, actions_n)

        # Define optimizer
        self._optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        
        # Define the losses (most likely some `torch.nn.*Loss`).
        self._loss = torch.nn.MSELoss()
        
        self.to(self.device)


    def forward(self, x):
        if x.dim() == 3: # # Accept both a single state (H, W, C) and a batch (N, H, W, C).
            x = x.unsqueeze(0)

        x = x.permute(0, 3, 1, 2).float() / 255.0 # Input: (N, H, W, C) -> (N, C, H, W)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # x = self.relu(self.conv4(x))

        x = torch.flatten(x, 1)  # (N, 64, 5, 5) -> (N, 1600)

        x = self._final_linear(x)
        
        return x
    
    # Define a training method. 
    @npfl139.typed_torch_function(device, torch.float32, torch.float32)
    def train_step(self, states: torch.Tensor, q_values: torch.Tensor) -> None:
        self.train()
        predictions = self(states)
        loss = self._loss(predictions, q_values)

        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self(states)

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: "Network") -> None:
        self.load_state_dict(other.state_dict())

def choose_action(env, epsilon, q_values):
    if np.random.rand() < epsilon:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = np.argmax(q_values)
    return action

def evaluate_training(eval_env, model, eval_type, evaluation_episodes, target_value = 450):
    returns = []
    for _ in range(evaluation_episodes):
        state, done = eval_env.reset()[0], False

        episode_return = 0
        while not done:
            # Choose a greedy action
            action = np.argmax(model.predict(state[np.newaxis])[0])
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_return += reward

        returns.append(episode_return)

    mean_return = np.mean(returns)

    if (eval_type == "long"):
        print("Evaluation return:", mean_return)
        if mean_return > target_value:
            print("Target reached, stopping training.")
            return True, mean_return
    
    return False, mean_return

def load_last_return(txt_path: str) -> float:
    if not os.path.exists(txt_path):
        return 0.0

    last_value = 0.0

    with open(txt_path, "r") as f:
        lines = f.readlines()

    # Iterate from bottom → top
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue

        # Works for both formats:
        # "mean_return: 135.6200"
        # "episode: 10, mean_return: 127.2400"
        if "mean_return:" in line:
            try:
                value = float(line.split("mean_return:")[1].strip())
                return value
            except ValueError:
                continue

    return last_value

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Create evaluation environment
    eval_env = npfl139.EvaluationEnv(gym.make("npfl139/CartPolePixels-v1"), args.seed, args.render_each, evaluate_for=30, report_each=0)

    # Resize envs
    env = gym.wrappers.ResizeObservation(env, (64, 64))
    eval_env = gym.wrappers.ResizeObservation(eval_env, (64, 64))

    # Construct the online network
    network = Network(env, args)

    # Construct the target network
    target_network = Network(env, args)
    target_network.copy_weights_from(network)

    best_return = 0
    if args.load_model and os.path.exists(args.pretrained_model_path):
        print("Loading pretrained model...")
        network.load_state_dict(torch.load(args.pretrained_model_path, map_location=network.device))
        target_network.copy_weights_from(network)

        best_return = load_last_return("best_model.txt")
        print(f"Loaded best_return = {best_return:.4f}")

    # Replay memory; the `max_length` parameter is its maximum capacity.
    replay_buffer = npfl139.ReplayBuffer(max_length=100_000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent
        ...

        # Final evaluation
        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False
            while not done:
                # TODO: Choose a greedy action.
                action = ...
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # TODO: Perform training
    epsilon = args.epsilon
    training = True
    train_steps = 0
    episode = 0
    while training:
        # Perform episode
        state, done = env.reset()[0], False

        while not done:
            # Epsilon-greedy action selection.
            q_values = network.predict(state[np.newaxis])[0]
            action = choose_action(env, epsilon, q_values)

            # Make a step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Append state, action, reward, done and next_state to replay_buffer
            # replay_buffer.append(Transition(state, action, reward, done, next_state))
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            if len(replay_buffer) > args.batch_size*20:
                samples = replay_buffer.sample(args.batch_size, replace=True)

                # ------ Double Deep Q Network ------
                q_values = network.predict(samples.state)

                next_q_online = network.predict(samples.next_state)
                next_actions = np.argmax(next_q_online, axis=1)

                next_q_target = target_network.predict(samples.next_state)
                next_q_selected = next_q_target[np.arange(args.batch_size), next_actions]

                targets = samples.reward + args.gamma * next_q_selected * (~samples.done)

                q_values[np.arange(args.batch_size), samples.action] = targets
                network.train_step(samples.state, q_values)

                train_steps += 1
                if train_steps % args.target_update_freq == 0:
                    print("Weights updated")
                    target_network.copy_weights_from(network)
            
            state = next_state

        episode += 1

        # Evaluate regularly and stop once the target is reached.
        if episode % 10 == 0:
            _, short_mean = evaluate_training(eval_env, network, "short", args.evaluation_episodes_short, target_value = 450)

            if (best_return < short_mean):
                target_reached, long_mean = evaluate_training(eval_env, network, "long", args.evaluation_episodes_long, target_value = 450)
                if long_mean > best_return:
                    best_return = long_mean
                    target_network.copy_weights_from(network)   # only if you want to keep the best weights there
                    torch.save(network.state_dict(), "best_model.pt")
                    with open("best_model.txt", "a") as f:
                        f.write(f"episode: {episode}, mean_return: {best_return:.4f}\n")
                    print(f"New best model saved, mean return = {best_return}")

                if (target_reached or args.max_episodes < episode): 
                    break # Finish training if target reached 
            
            if (args.max_episodes < episode):
                break # Finish training if max allowed number of episodes exceeded
            
        
        if args.epsilon_final_at:
            epsilon = np.interp(episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
        




if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("npfl139/CartPolePixels-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)