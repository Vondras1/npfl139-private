#!/usr/bin/env python3
import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from typing import Tuple

import npfl139

npfl139.require_version("2526.6")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.

parser.add_argument("--policy_hidden_layer_size", default=64, type=int, help="Size of policy hidden layer.")
parser.add_argument("--value_hidden_layer_size", default=64, type=int, help="Size of value hidden layer.")
parser.add_argument("--learning_rate", default=0.0001, type=float)
parser.add_argument("--entropy_coef", default=0.01, type=float)
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")

parser.add_argument("--evaluation_each", default=10, type=int, help="Evaluate every N training iterations.")
parser.add_argument("--evaluation_episodes_short", default=20, type=int, help="Number of evaluation episodes.")
parser.add_argument("--evaluation_episodes_long", default=200, type=int, help="Number of evaluation episodes.")
parser.add_argument("--target_return", default=500, type=float, help="Target mean return.")
parser.add_argument("--model_path", default="best_model.pt", type=str, help="Path to saved model.")
parser.add_argument("--load_model", default=False, action="store_true", help="Load pretrained model if it exists.")

WHITE = np.array((255, 255, 255))
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)

def __get_highest_pole_point(image: np.ndarray) -> Tuple[int, int]:
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if np.array_equal(image[y, x], WHITE):
                return y, x
    raise ValueError("Pole not found in the image.")


def __get_base_pixel(image: np.ndarray) -> int:
    BASE_Y = 55
    for x in range(image.shape[1]):
        if np.array_equal(image[BASE_Y, x], WHITE):
            return BASE_Y, x

    raise ValueError("Pole not found in the image.")


def __get_left_base_pixel(image: np.ndarray) -> int:
    BASE_Y = 62
    for x in range(image.shape[1]):
        if np.array_equal(image[BASE_Y, x], GRAY):
            return x
    raise ValueError("Base not found in the image.")


def _get_angle(image: np.ndarray) -> float:
    pole_point = __get_highest_pole_point(image)
    base_point = __get_base_pixel(image)

    dy = pole_point[0] - base_point[0]
    dx = pole_point[1] - base_point[1]

    angle = np.arctan2(dy, dx)
    return angle


class Network(torch.nn.Module):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.args = args

        actions_n = env.action_space.n
        H, W, C = env.observation_space.shape  # It should be [64x64x3]

        self.relu = torch.nn.ReLU()

        self.decoder_linear = torch.nn.Linear(2, 128)

        # Define two heads. One for policy, second for value function
        self._policy = torch.nn.Sequential(
            torch.nn.Linear(128, args.policy_hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.policy_hidden_layer_size, actions_n),
        ).to(self.device)

        self._value = torch.nn.Sequential(
            torch.nn.Linear(128, args.value_hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.value_hidden_layer_size, 1),
        ).to(self.device)

        # Define optimizer
        self._optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)

        # Define the losses (most likely some `torch.nn.*Loss`).
        self._policy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self._value_loss = torch.nn.MSELoss()

        # Scheduler
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=self.args.episodes,
            eta_min=0.0000001)

        self.to(self.device)

    def forward(self, x: torch.Tensor):
        x = x.float()
        x = self.relu(self.decoder_linear(x))

        policy = self._policy(x)
        value = self._value(x)

        return policy, value

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train_step(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        policy, value = self(states)

        # value loss
        value = value.squeeze(-1)
        value_loss = self._value_loss(value, returns)

        # delta (baseline)
        delta = (returns - value).detach()

        # policy loss
        policy_losses = self._policy_loss(policy, actions)
        policy_loss = (delta * policy_losses).mean()

        # # Prevent premature convergence
        dist = torch.distributions.Categorical(logits=policy)
        entropy = dist.entropy().mean()

        # Total loss
        # print(f"value_loss = {value_loss}, policy_loss = {policy_loss}, entropy = {entropy}")
        loss = (0.1) * value_loss + policy_loss - 0.01 * entropy

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._scheduler.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Define the prediction method returning policy probabilities.
        logits, _ = self(states)  # [num_actions]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()  # Return as NumPy array


def evaluate_model(eval_env, model, eval_type, evaluation_episodes, target_value=450):
    returns = []
    for _ in range(evaluation_episodes):
        state, done = eval_env.reset()[0], False

        episode_return = 0
        while not done:
            pole_angle = _get_angle(state)
            base_pixel = __get_left_base_pixel(state)
            # Choose a greedy action
            features = np.array([pole_angle, base_pixel], dtype=np.float32)
            action = np.argmax(model.predict(features[None])[0])
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

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if "mean_return:" in line:
            try:
                return float(line.split("mean_return:")[1].strip())
            except ValueError:
                continue

    return 0.0


def load_ensemble_agents(env, args):
    # model paths
    lst = os.listdir("./cart_pole_models")
    model_paths = []
    for model in lst:
        model_paths.append(f"./cart_pole_models/{model}/best_model.pt")

    agents = []
    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model file: {path}")

        agent = Network(env, args)
        agent.load_state_dict(torch.load(path, map_location=agent.device))
        agent.eval()
        agents.append(agent)

    print(f"Loaded {len(agents)} model(s) for ensemble:")
    for path in model_paths:
        print(f"  - {path}")

    return agents


def ensemble_predict(agents, state: np.ndarray) -> np.ndarray:
    probs = []
    for agent in agents:
        agent_probs = agent.predict(state)[0]  # shape [actions]
        probs.append(agent_probs)

    mean_probs = np.mean(probs, axis=0)
    return mean_probs


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Create evaluation environment
    eval_env = npfl139.EvaluationEnv(gym.make("npfl139/CartPolePixels-v1"), args.seed, args.render_each,
                                     evaluate_for=30, report_each=0)

    # Resize envs
    env = gym.wrappers.ResizeObservation(env, (64, 64))
    eval_env = gym.wrappers.ResizeObservation(eval_env, (64, 64))

    # Construct the agent.
    agent = Network(env, args)

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent
        agents = load_ensemble_agents(env, args)
        # Final evaluation
        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False
            pole_angle = _get_angle(state)
            base_pixel = __get_left_base_pixel(state)
            while not done:
                # TODO: Choose a greedy action.
                ensemble_predicted_probs = ensemble_predict(agents, state)
                action = int(np.argmax(ensemble_predicted_probs))
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    best_return = 0.0
    log_path = os.path.splitext(args.model_path)[0] + ".txt"

    if args.load_model and os.path.exists(args.model_path):
        print(f"Loading pretrained model from {args.model_path}")
        agent.load_state_dict(torch.load(args.model_path, map_location=agent.device))
        best_return = load_last_return(log_path)
        print(f"Loaded last logged mean return = {best_return:.4f}")

    # Perform training
    for episode in range(1, args.episodes + 1):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                # Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                pole_angle = _get_angle(state)
                base_pixel = __get_left_base_pixel(state)
                print(f"Pole angle: {pole_angle:.4f}, Base pixel: {base_pixel}")

                # custom_state = torch.Tensor([pole_angle, base_pixel])
                custom_state = np.array([pole_angle, base_pixel], dtype=np.float32)

                # policy_probs = agent.predict(custom_state)
                # action = np.random.choice(len(policy_probs), p=policy_probs)
                policy_probs = agent.predict(custom_state[None])[0]
                action = np.random.choice(len(policy_probs), p=policy_probs)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(custom_state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # Compute returns by summing rewards (with discounting)
            G = 0
            returns = []

            for r in reversed(rewards):
                G = r + args.gamma * G
                returns.insert(0, G)

            # Add states, actions and returns to the training batch
            batch_states += states
            batch_actions += actions
            batch_returns += returns

        # Train using the generated batch.
        agent.train_step(np.array(batch_states), np.array(batch_actions), np.array(batch_returns))

        # Evaluation
        if episode % args.evaluation_each == 0:
            _, short_mean = evaluate_model(eval_env, agent, "short", args.evaluation_episodes_short, args.target_return)

            if short_mean > 450 or episode % 50 == 0:  # episode % 100 == 0 ---> Provide feedback even if you didn't improve your score
                target_reached, long_mean = evaluate_model(eval_env, agent, "long", args.evaluation_episodes_long, args.target_return)
                if long_mean > best_return-15:
                    best_return = long_mean
                    torch.save(agent.state_dict(), args.model_path)

                    with open(log_path, "a") as f:
                        f.write(f"episode: {episode}, mean_return: {best_return:.4f}\n")

                    print(f"New best model saved, mean return = {best_return:.4f}")

                if target_reached:
                    print("Target reached, stopping training.")
                    break

    torch.save(agent.state_dict(), "final_model.pt")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("npfl139/CartPolePixels-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)