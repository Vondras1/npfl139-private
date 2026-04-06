#!/usr/bin/env python3
import argparse
import json

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.6")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="LunarLander-v3", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=32, type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0224, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="paac_actor.pt", type=str, help="Path to the actor model.")


class Agent:
    device = torch.device("cpu")
    # Use the following line instead to use GPU if available.
    # device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Similarly to reinforce with baseline, define two components:
        # - an actor, which predicts distribution over the actions, and
        # - a critic, which predicts the value function.
        #
        # Use independent networks for both of them, each with
        # `args.hidden_layer_size` neurons in one ReLU hidden layer,
        # and train them using Adam with given `args.learning_rate`.
        self._policy = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
        ).to(self.device)

        self._value = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1),
        ).to(self.device)

        self.args = args

        # TODO: Define an optimizer. Using `torch.optim.Adam` optimizer with
        # the given `args.learning_rate` is a good default.
        self._policy_optimizer = torch.optim.Adam(self._policy.parameters(), lr=args.learning_rate)
        
        self._value_optimizer = torch.optim.Adam(self._value.parameters(), lr=args.learning_rate)

        # TODO: Define the loss (most likely some `torch.nn.*Loss`).
        self._policy_loss = torch.nn.CrossEntropyLoss(reduction="none")

        self._value_loss = torch.nn.MSELoss()

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Train the policy network using policy gradient theorem
        # and the value network using MSE.
        #
        # The `args.entropy_regularization` might be used to include actor
        # entropy regularization -- the assignment can be solved even without
        # it, but my reference solution learns quicklier when using it.
        # In any case, `torch.distributions.Categorical` is a suitable distribution
        # offering the `.entropy()` method.
        policy = self._policy(states)
        value = self._value(states)

        # Prevent premature convergence
        dist = torch.distributions.Categorical(logits=policy)
        entropy = dist.entropy().mean()

        # value loss
        value = value.squeeze(-1)
        value_loss = self._value_loss(value, returns)

        # delta (baseline)
        delta = (returns - value).detach() 

        # policy loss
        policy_losses = self._policy_loss(policy, actions)
        policy_loss = (delta*policy_losses).mean() - self.args.entropy_regularization * entropy

        # UPDATE POLICY
        self._policy_optimizer.zero_grad() # Zero gradients for every batch!
        policy_loss.backward() # gradients
        self._policy_optimizer.step() # Adjust learning weights

        # UPDATE VALUE
        self._value_optimizer.zero_grad() # Zero gradients for every batch!
        value_loss.backward() # gradients
        self._value_optimizer.step() # Adjust learning weights

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted action probabilities.
        logits = self._policy(states)  # [num_actions]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy() # Return as NumPy array

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return estimates of the value function.
        value_fce = self._value(states).squeeze(-1).detach().cpu().numpy()
        return value_fce

    # Serialization methods.
    def save_actor(self, path: str) -> None:
        torch.save(self._policy.state_dict(), path)

    def load_actor(self, path: str) -> None:
        self._policy.load_state_dict(torch.load(path, map_location=self.device))

    @staticmethod
    def save_args(path: str, args: argparse.Namespace) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(vars(args), file, ensure_ascii=False, indent=2)

    @staticmethod
    def load_args(path: str) -> argparse.Namespace:
        with open(path, "r", encoding="utf-8-sig") as file:
            args = json.load(file)
        return argparse.Namespace(**args)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    agent = Agent(env, args if not args.recodex else Agent.load_args(args.model_path + ".json"))

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # TODO: Predict an action using the greedy policy.
            predicted_probs = agent.predict_actions(state)
            action = np.argmax(predicted_probs)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # ReCodEx evaluation.
    if args.recodex:
        agent.load_actor(args.model_path)
        while True:
            evaluate_episode(start_evaluation=True)

    # Create the vectorized environment, using the SAME_STEP autoreset mode.
    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC,
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    states = vector_env.reset(seed=args.seed)[0]

    training = True
    best_return = 0
    target_return = 260
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Choose actions using `agent.predict_actions`.
            action_probs = agent.predict_actions(states)
            actions = np.array([np.random.choice(len(probs), p=probs) for probs in action_probs], dtype=np.int64)

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = terminated | truncated

            # TODO: Compute estimates of returns by one-step bootstrapping
            next_values = agent.predict_values(next_states)
            returns = rewards + args.gamma * (~dones) * next_values

            # TODO: Train agent using current states, chosen actions and estimated returns.
            agent.train(states, actions, returns)

            states = next_states

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        avg_return = np.mean(returns)
        print(f"Mean return {np.mean(returns)}")
        print("-----------")


        if avg_return > best_return:
            longer_returns = [evaluate_episode() for _ in range(5*args.evaluate_for)]
            avg_long_return = np.mean(longer_returns)
            if avg_long_return > best_return:
                agent.save_actor(args.model_path)
                agent.save_args(args.model_path + ".json", args)
                print("returns_best:", best_return, "\t actor->saved") 


    # Save the agent
    agent.save_actor(args.model_path)
    agent.save_args(args.model_path + ".json", args)

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)
