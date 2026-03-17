#!/usr/bin/env python3
# Team:
# 1ac5d633-f96f-42a3-846d-31bcb01d041f
# 9fafb47f-e1c5-4d7c-8ce5-8a6f5bdcd751

import argparse
import collections
import os
import re
import json

import gymnasium as gym
import numpy as np
import torch

from gymnasium.wrappers.vector import ResizeObservation as VectorResizeObservation
from gymnasium.wrappers.vector import GrayscaleObservation as VectorGrayscaleObservation

import npfl139
npfl139.require_version("2526.4")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=5, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--continuous", default=1, type=int, help="Use continuous actions.")
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=200, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=1000, type=int, help="Target update frequency.")
parser.add_argument("--evaluation_episodes", default=50, type=int, help="Number of evaluation episodes.")
parser.add_argument("--num_envs", default=8, type=int, help="Number of parallel environments.")
parser.add_argument("--max_episodes", default=1000, type=int, help="Maximum number of episodes.")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actions = [
    [-0.5, 0.0, 0.0],   # soft left
    [-0.8, 0.0, 0.0],   # left
    [ 0.5, 0.0, 0.0],   # soft right
    [ 0.8, 0.0, 0.0],   # right
    [ 0.0, 0.6, 0.0],   # soft gas
    [ 0.0, 1.0, 0.0],   # gas
    [ 0.0, 0.0, 0.5],   # soft brake
    [ 0.0, 0.0, 0.8],   # brake
]
actions_n = len(actions)

class AgentSaver:
    def __init__(self, args, main_folder, model_name = "q_model_racing.pt"):
        self.args = args
        self.recodex = args.recodex
        self.main_folder = main_folder
        self.model_name = model_name
        self.args_name = "args.json"

    def save_next_agent(self, model) -> None:
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

        self.save_agent(new_folder, model)

        print(f"Agent saved to {new_folder}")

    def save_agent(self, folder_path: str, model) -> None:
        """Save model weights and parser arguments into one folder."""

        os.makedirs(folder_path, exist_ok=True)

        model_path = os.path.join(folder_path, self.model_name)
        args_path = os.path.join(folder_path, self.args_name)

        torch.save(model.state_dict(), model_path)

        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, indent=4, ensure_ascii=False)

    def load_agent(self, model, agent_name: str) -> tuple[object, dict | None]:
        """Load trained model and saved parser arguments from one folder."""

        if self.recodex:
            model_path = "q_model_racing.pt"
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            return model, None

        model_path = os.path.join(self.main_folder, agent_name, self.model_name)
        args_path = os.path.join(self.main_folder, agent_name, self.args_name)

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        with open(args_path, "r", encoding="utf-8") as f:
            saved_args = json.load(f)

        return model, saved_args

def choose_action(env, epsilon, q_values, actions_n):
    if np.random.rand() < epsilon:
        action = np.random.randint(0, actions_n)
    else:
        action = np.argmax(q_values)
    return action

def make_stacked_state(frame_buffer):
    return np.concatenate(list(frame_buffer), axis=-1)

def evaluate_training(eval_env, model, args, target_value = 500):
    returns = []
    for _ in range(args.evaluation_episodes):
        state, done = eval_env.reset()[0], False

        frame_buffer = collections.deque(maxlen=4)
        for _ in range(4):
            frame_buffer.append(state)
        stacked_state = make_stacked_state(frame_buffer)

        episode_return = 0
        while not done:
            # Choose a greedy action
            # action = np.argmax(model.predict(state[np.newaxis])[0])
            action = np.argmax(model.predict(stacked_state[np.newaxis])[0])
            state, reward, terminated, truncated, _ = eval_env.step(actions[action])
            done = terminated or truncated
            episode_return += reward

            frame_buffer.append(state)
            stacked_state = make_stacked_state(frame_buffer)

        returns.append(episode_return)

    mean_return = np.mean(returns)
    print("Evaluation return:", mean_return)
    if mean_return > target_value:
        print("Target reached, stopping training.")
        return True

    return False, mean_return

class Network(torch.nn.Module):
    def __init__(self, env, args, actions_n):
        super().__init__()
        self.env = env
        self.args = args

        # action_num = env.action_space.n
        H, W, C = env.observation_space.shape

        in_channels = C * 4 #(4 stacked images to gain access to speed)
        out_channels = 32
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = torch.nn.ReLU()
        # 32 x 32 x 32

        in_channels = out_channels
        out_channels = 64
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        # 16 x 16 x 64

        in_channels = out_channels
        out_channels = 128
        self.conv3 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        # 4 x 4 x 128

        # in_channels = out_channels
        # out_channels = 256
        # self.conv4 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        # self.batch_norm4 = torch.nn.BatchNorm2d(out_channels)
        # self.relu = torch.nn.ReLU()
        # 4 x 4 x 256

        # pool
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))

        # Linear
        self.fc1 = torch.nn.Linear(out_channels, 128)
        self.fc2 = torch.nn.Linear(128, actions_n)

        self._optimizer = torch.optim.AdamW(self.parameters(), lr=args.learning_rate)
        self._loss = torch.nn.MSELoss()
        self.to(DEVICE)


    def forward(self, x):
        # Input: (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2).float() / 255.0

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        # (N, 128, 4, 4)
        x = self.pool(x)
        # (N, 128, 1, 1)
        x = torch.flatten(x, 1)
        # (N, 128)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, and include the index of the action to which
    #   the new q_value belongs
    # The code below implements the first option, but you can change it if you want.
    #
    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(DEVICE, torch.float32, torch.float32)
    def train_step(self, states: torch.Tensor, q_values: torch.Tensor) -> None:
        self.train()
        predictions = self(states)
        loss = self._loss(predictions, q_values)

        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @npfl139.typed_torch_function(DEVICE, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self(states)

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: "Network") -> None:
        self.load_state_dict(other.state_dict())

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    eval_env = npfl139.EvaluationEnv(
        gym.make("npfl139/CarRacingFS-v3", frame_skip=args.frame_skip, continuous=args.continuous),
        args.seed, args.render_each, evaluate_for=15, report_each=1)
    
    # If you want, you can wrap even the `npfl139.EvaluationEnv` with additional wrappers, like
    #   env = gym.wrappers.ResizeObservation(env, (64, 64))
    # or
    #   env = gym.wrappers.GrayscaleObservation(env)
    # However, if you do that, you can no longer call just `env.reset(start_evaluation=True)`;
    # instead, you need to pass the `start_evaluation` to the inner environment using
    #   env.reset(options={"start_evaluation": True})
    env = gym.wrappers.ResizeObservation(env, (64, 64))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)

    eval_env = gym.wrappers.ResizeObservation(eval_env, (64, 64))
    eval_env = gym.wrappers.GrayscaleObservation(eval_env, keep_dim=True)

    # Construct the network
    network = Network(env, args, actions_n)

    # Construct the target network
    target_network = Network(env, args, actions_n)
    target_network.copy_weights_from(network)

    # Replay memory; the `max_length` parameter is its maximum capacity.
    replay_buffer = npfl139.ReplayBuffer(max_length=1_000_000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon

    # Create agent latter used for loading and saving models
    agent = AgentSaver(args, "04/racing", model_name = "q_model_racing.pt")

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent

        model, _ = agent.load_agent(network, None)

        # Final evaluation
        while True:
            # state, done = env.reset(start_evaluation=True)[0], False
            state, done = env.reset(options={"start_evaluation": True})[0], False

            frame_buffer = collections.deque(maxlen=4)
            for _ in range(4):
                frame_buffer.append(state)
            stacked_state = make_stacked_state(frame_buffer)

            while not done:
                # TODO: Choose a greedy action
                action = np.argmax(model.predict(stacked_state[np.newaxis])[0])
                state, reward, terminated, truncated, _ = env.step(actions[action])
                done = terminated or truncated
                frame_buffer.append(state)
                stacked_state = make_stacked_state(frame_buffer)

    # TODO: Implement a suitable RL algorithm and train the agent.
    #
    # If you want to create N multiprocessing parallel environments, use
    #   vector_env = gym.make_vec("npfl139/CarRacingFS-v3", N, gym.VectorizeMode.ASYNC,
    #                             frame_skip=args.frame_skip, continuous=args.continuous)
    #   vector_env.reset(seed=args.seed)  # The individual environments get incremental seeds
    #
    # There are several Autoreset modes available, see https://farama.org/Vector-Autoreset-Mode.
    # To change the autoreset mode to SAME_STEP from the default NEXT_STEP, pass
    #   vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
    # as an additional argument to the above `gym.make_vec`.
    training = True
    train_steps = 0
    episode = 0
    while training:
        # Perform episode
        state, done = env.reset()[0], False

        frame_buffer = collections.deque(maxlen=4)
        for _ in range(4):
            frame_buffer.append(state)
        stacked_state = make_stacked_state(frame_buffer)

        while not done:
            # Choose an action.
            # q_values = network.predict(state[np.newaxis])[0]
            q_values = network.predict(stacked_state[np.newaxis])[0]
            action = choose_action(env, epsilon, q_values, actions_n)

            # Make a step
            next_state, reward, terminated, truncated, _ = env.step(actions[action])
            done = terminated or truncated

            frame_buffer.append(next_state)
            stacked_next_state = make_stacked_state(frame_buffer)

            # Append state, action, reward, done and next_state to replay_buffer
            # replay_buffer.append(Transition(state, action, reward, done, next_state))
            replay_buffer.append(Transition(stacked_state, action, reward, done, stacked_next_state))

            if len(replay_buffer) > args.batch_size*10:
                samples = replay_buffer.sample(args.batch_size, replace=True)

                # ------ Deep Q Network ------
                # q_values = network.predict(samples.state)
                # q_values_next = target_network.predict(samples.next_state)

                # targets = samples.reward + args.gamma * np.max(q_values_next, axis=1) * (~samples.done)

                # q_values[np.arange(args.batch_size), samples.action] = targets

                # network.train_step(samples.state, q_values)

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
                    print("Update")
                    target_network.copy_weights_from(network)

                # q_values = network.predict(samples.state)

                # next_q_online = network.predict(samples.next_state)
                # next_actions = np.argmax(next_q_online, axis=1)

                # next_q_target = target_network.predict(samples.next_state)
                # next_q_selected = next_q_target[np.arange(args.batch_size), next_actions]

                # targets = samples.reward + args.gamma * next_q_selected * (~samples.done)

                # q_values[np.arange(args.batch_size), samples.action] = targets
                # network.train_step(samples.state, q_values)

            state = next_state
            stacked_state = stacked_next_state

        episode += 1

        # evaluate and quit training if target reached
        if episode % 50 == 0:
            target_reached, mean_return = evaluate_training(eval_env, network, args, target_value = 900)
            if (target_reached or args.max_episodes < episode): 
                break # Finish training if target reached or max allowed number of episodes exceeded
            elif (mean_return > 750):
                agent.save_next_agent(network)


        if args.epsilon_final_at:
            epsilon = np.interp(episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

    # Save model
    agent.save_next_agent(network)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("npfl139/CarRacingFS-v3", frame_skip=main_args.frame_skip, continuous=main_args.continuous),
        main_args.seed, main_args.render_each, evaluate_for=15, report_each=1)

    main(main_env, main_args)
