#!/usr/bin/env python3
# Team
# 1ac5d633-f96f-42a3-846d-31bcb01d041f
# e0cfa255-0259-11eb-9574-ea7484399335
# 9fafb47f-e1c5-4d7c-8ce5-8a6f5bdcd751
import argparse
import copy
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
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size_policy", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--hidden_layer_size_value", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.04, type=float, help="Learning rate.")
parser.add_argument("--p_scheduler", default=0, type=float, help="Policy scheduler.")
parser.add_argument("--v_scheduler", default=0, type=float, help="Value scheduler.")

class Agent:
    device = torch.device("cpu")
    # Use the following line instead to use GPU if available.
    # device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model of the policy and the value networks.
        #
        # In addition to the policy network defined in the `reinforce` assignment,
        # you need a value network for computing the baseline. It can be for example
        # another independent model with a single hidden layer and an output layer
        # with a single output and no activation. You can also experiment with just
        # a single shared model with two heads (the policy head and the value head),
        # but such a model is more difficult to train because of possible different
        # scales of the two losses.
        #
        # Using Adam optimizer with given `args.learning_rate` for both models
        # is a good default.
        self._policy = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size_policy),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size_policy, env.action_space.n),
        ).to(self.device)

        self._value = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size_value),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size_value, 1),
        ).to(self.device)

        self.args = args

        # TODO: Define an optimizer. Using `torch.optim.Adam` optimizer with
        # the given `args.learning_rate` is a good default.
        self._policy_optimizer = torch.optim.Adam(self._policy.parameters(), lr=args.learning_rate)
        
        self._value_optimizer = torch.optim.Adam(self._value.parameters(), lr=args.learning_rate)

        # TODO: Define the loss (most likely some `torch.nn.*Loss`).
        self._policy_loss = torch.nn.CrossEntropyLoss(reduction="none")

        self._value_loss = torch.nn.MSELoss()

        if self.args.p_scheduler:
            # self._policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #                         self._policy_optimizer,
            #                         T_max=self.args.episodes // self.args.batch_size,
            #                         eta_min=0.008)
            self._policy_scheduler = torch.optim.lr_scheduler.LinearLR(
                self._policy_optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.args.episodes // self.args.batch_size
            )
            
        if self.args.v_scheduler:
            # self._value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #                         self._value_optimizer,
            #                         T_max=self.args.episodes // self.args.batch_size,
            #                         eta_min=0.008)
            self._value_scheduler = torch.optim.lr_scheduler.LinearLR(
                self._policy_optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.args.episodes // self.args.batch_size
            )
            

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Define the training method.
        #
        # You should:
        # - compute the predicted baseline using the baseline model
        # - train the policy model, using `returns - predicted_baseline` as
        #   advantage estimate
        # - train the baseline model to predict `returns`
        
        # value loss
        value = self._value(states).squeeze(-1) # Instead of predicting returns in range 0-500 predict it only in range 0-1. Should be easir for network
        value_loss = self._value_loss(value, returns)

        # delta (baseline)
        delta = (returns - value).detach()

        # policy loss
        policy = self._policy(states)
        policy_losses = self._policy_loss(policy, actions)
        policy_loss = (delta*policy_losses).mean()

        # BACK PROPAGATION - UPDATE POLICY
        self._policy_optimizer.zero_grad() # Zero gradients for every batch!
        policy_loss.backward() # gradients
        self._policy_optimizer.step() # Adjust learning weights
        if self.args.p_scheduler:
            self._policy_scheduler.step()

        # BACK PROPAGATION - UPDATE VALUE
        self._value_optimizer.zero_grad() # Zero gradients for every batch!
        value_loss.backward() # gradients
        self._value_optimizer.step() # Adjust learning weights
        if self.args.v_scheduler:
            self._value_scheduler.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Define the prediction method returning policy probabilities.
        logits = self._policy(states)  # [num_actions]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy() # Return as NumPy array


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    agent = Agent(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                # TODO(reinforce): Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                policy_probs = agent.predict(state)
                action = np.random.choice(len(policy_probs), p = policy_probs)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute returns by summing rewards (with discounting)
            G = 0
            returns = []

            for r in reversed(rewards):
                G = r + args.gamma * G
                returns.insert(0, G)

            # TODO(reinforce): Add states, actions and returns to the training batch
            batch_states += states
            batch_actions += actions
            batch_returns += returns

        # TODO(reinforce): Train using the generated batch.
        agent.train(np.array(batch_states), np.array(batch_actions), np.array(batch_returns))

    # Final evaluation
    if args.recodex:
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # TODO(reinforce): Choose a greedy action.
                probs = agent.predict(state)
                action = np.argmax(probs)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    
    else:
        # Final evaluation (finite number of episodes)
        evaluation_episodes = 100
        returns = []

        for _ in range(evaluation_episodes):
            state, done = env.reset(start_evaluation=False)[0], False
            total_reward = 0

            while not done:
                probs = agent.predict(state)
                action = np.argmax(probs)

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

            returns.append(total_reward)

        mean_return = np.mean(returns)
        
        return mean_return


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    learning_rates = [0.005, 0.007, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1]
    batch_sizes = [6, 7, 8, 9, 10, 11, 12, 14, 16, 32, 64]
    lr_decay = [0, 1]

    results = []

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for decay in lr_decay:
                means = []

                for seed in [1, 2, 3, 4, 5, 40, 50, 60, 70, 80]:
                    args_copy = copy.deepcopy(main_args)  # avoid side effects
                    args_copy.learning_rate = lr
                    args_copy.batch_size = batch_size
                    args_copy.seed = seed
                    args_copy.p_scheduler = decay
                    args_copy.v_scheduler = decay

                    env = npfl139.EvaluationEnv(
                        gym.make("CartPole-v1"),
                        seed=seed,
                        render_each=0
                    )

                    mean = main(env, args_copy)
                    means.append(mean)

                mean = np.mean(means)

                results.append((lr, batch_size, decay, mean))

                print(f"lr={lr}, batch={batch_size}, decay={decay} -> {mean:.2f}")
    
    print("\n=== FINAL RESULTS ===")
    for lr, batch, decay, mean in results:
        print(f"{mean:7.2f} | lr={lr:<3} | decay={decay} | batch={batch}")
