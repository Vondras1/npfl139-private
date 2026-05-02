#!/usr/bin/env python3
import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.10")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="npfl139/SingleCollect-v0", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=0.2, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=8, type=int, help="Workers during experience collection.")
parser.add_argument("--epochs", default=4, type=int, help="Epochs to train each iteration.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--worker_steps", default=128, type=int, help="Steps for each worker to perform.")

# No need to tune this
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--model_path", default="ppo", type=str, help="Model path")
parser.add_argument("--load_model_path", default="ppo_545", type=str, help="Model path of pretrained model we want to load.")
parser.add_argument("--load_pretrained_models", default=False, action="store_true", help="Load pretrained models.")


class Agent:
    # Use GPU if available.
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self._args = args

        # TODO: Create an actor using a single hidden layer with `args.hidden_layer_size`
        # units and ReLU activation, produce a policy with `env.action_space.n` discrete actions.
        self._actor = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
        ).to(self.device)

        # TODO: Create a critic (value predictor) consisting of a single hidden layer with
        # `args.hidden_layer_size` units and ReLU activation, and an output layer with a single output.
        self._critic = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1),
        ).to(self.device)
    
        self._optimizer_actor = torch.optim.Adam(self._actor.parameters(), lr=args.learning_rate)
        self._optimizer_critic = torch.optim.Adam(self._critic.parameters(), lr=args.learning_rate)
        self._loss_fn = torch.nn.MSELoss()
        

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, action_probs: torch.Tensor,
              advantages: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Perform a single training step of the PPO algorithm.
        # For the policy model, the sum is the sum of:
        # - the PPO loss, where `self._args.clip_epsilon` is used to clip the probability ratio
        # - the entropy regularization with coefficient `self._args.entropy_regularization`.
        #   You can compute it for example using the `torch.distributions.Categorical` class.
        logits = self._actor(states)
        dist = torch.distributions.Categorical(logits=logits)

        new_action_probs = dist.log_prob(actions).exp()
        ratio = new_action_probs / action_probs

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - self._args.clip_epsilon, 1 + self._args.clip_epsilon) * advantages

        ppo_loss = -torch.mean(torch.minimum(unclipped, clipped))
        entropy_loss = -self._args.entropy_regularization * dist.entropy().mean()

        actor_loss = ppo_loss + entropy_loss

        self._optimizer_actor.zero_grad()
        actor_loss.backward()
        self._optimizer_actor.step()

        # TODO: The critic model is trained in a standard way, by using the MSE
        # error between the predicted value function and target returns.
        values = self._critic(states).squeeze(-1)
        critic_loss = torch.nn.functional.mse_loss(values, returns)

        self._optimizer_critic.zero_grad()
        critic_loss.backward()
        self._optimizer_critic.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted action probabilities.
        probs = torch.softmax(self._actor(states), dim=-1)
        return probs

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return estimates of value function.
        values = self._critic(states).squeeze(-1)
        return values

    def save_models(self, path: str) -> None:
        torch.save({
            "actor": self._actor.state_dict(),
            "critic1": self._critic.state_dict(),
        }, path)
    
    def load_models(self, path: str) -> None:
        model_blocks = torch.load(path, map_location=self.device)

        self._actor.load_state_dict(model_blocks["actor"])
        self._critic.load_state_dict(model_blocks["critic1"])

    @staticmethod
    def save_args(path: str, args: argparse.Namespace) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(vars(args), file, ensure_ascii=False, indent=2)

    @staticmethod
    def load_args(path: str) -> argparse.Namespace:
        with open(path, "r", encoding="utf-8-sig") as file:
            args = json.load(file)
        return argparse.Namespace(**args)

def extract_score_from_model_path(path: str) -> float | None:
    filename = os.path.basename(path)
    parts = filename.split("_")
    if not parts:
        return None

    last_part = parts[-1]
    try:
        return float(last_part)
    except ValueError:
        return None

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    agent = Agent(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # TODO: Predict an action by using a greedy policy.
            probs = agent.predict_actions(state)
            action = np.argmax(probs)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create an asynchronous vector environment for training.
    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC,
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})

    best_return = -100
    if args.load_pretrained_models or args.recodex:
        agent.load_models(args.load_model_path)
        score = extract_score_from_model_path(args.load_model_path)
        if score is not None:
            best_return = score
            print(f"Loaded best_return: {best_return}")

    # Training
    state = vector_env.reset(seed=args.seed)[0]
    training, iteration = True, 0
    while training:
        # Collect experience. Notably, we collect the following quantities
        # as tensors with the first two dimensions `[args.worker_steps, args.envs]`.
        states, actions, action_probs, rewards, dones, values = [], [], [], [], [], []
        for _ in range(args.worker_steps):
            # TODO: Choose `action`, which is a vector of `args.envs` actions, each
            # sampled from the corresponding policy generated by the `agent.predict`
            # executed on the vector `state`.
            policy = agent.predict_actions(state)          # shape [envs, actions]
            dist = torch.distributions.Categorical(probs=torch.tensor(policy))
            action = dist.sample().numpy()                 # shape [envs]

            # # In case the previous does not work, it is also possible to use
            # action = np.array([
            #     np.random.choice(env.action_space.n, p=policy[i])
            #     for i in range(args.envs)
            # ])

            # Perform the environment interaction.
            next_state, reward, terminated, truncated, _ = vector_env.step(action)
            done = terminated | truncated

            # TODO: Compute and collect the required quantities.
            value = agent.predict_values(state)

            # Selects the probability of the action that was actually sampled. 
            # These are old action probabilities, which we will use for importance sampling in the PPO loss.
            selected_action_probs = policy[np.arange(args.envs), action] 

            states.append(state)          # where action was chosen
            actions.append(action)        # chosen action
            action_probs.append(selected_action_probs)      # old probability of that action
            rewards.append(reward)        # reward after action
            dones.append(done)            # whether episode ended
            values.append(value)          # V(state)

            state = next_state


        states = np.asarray(states)
        actions = np.asarray(actions)
        action_probs = np.asarray(action_probs)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        values = np.asarray(values)

        # TODO: Estimate `advantages` and `returns` (they differ only by the value function estimate)
        # using lambda-return with coefficients `args.trace_lambda` and `args.gamma`.
        # You need to handle both the cases that (a) the last episode is probably unfinished, and
        # (b) there are multiple episodes in the collected data.
        advantages = np.zeros_like(rewards)
        last_advantage = np.zeros(args.envs)
        last_values = agent.predict_values(state)   # state je už next_state po loopu

        for t in reversed(range(args.worker_steps)):
            if t == args.worker_steps - 1:
                next_value = last_values
            else:
                next_value = values[t + 1]

            delta = rewards[t] + args.gamma * next_value * (1 - dones[t]) - values[t]

            advantages[t] = delta + args.gamma * args.trace_lambda * (1 - dones[t]) * last_advantage

            last_advantage = advantages[t]
        
        returns = advantages + values

        # TODO: Train for `args.epochs` using the collected data. In every epoch,
        # you should randomly sample batches of size `args.batch_size` from the collected data.
        # A possible approach is to create a dataset of `(states, actions, action_probs, advantages, returns)`
        # quintuples using a single `torch.utils.data.StackDataset` and then use a dataloader.

        # only now flatten
        states = states.reshape((-1, *states.shape[2:]))
        actions = actions.flatten()
        action_probs = action_probs.flatten()
        advantages = advantages.flatten()
        returns = returns.flatten()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(action_probs),
            torch.from_numpy(advantages),
            torch.from_numpy(returns),
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True
        )        

        for _ in range(args.epochs):
            for batch in loader:
                b_states, b_actions, b_action_probs, b_advantages, b_returns = batch

                agent.train(
                    b_states.numpy(),
                    b_actions.numpy(),
                    b_action_probs.numpy(),
                    b_advantages.numpy(),
                    b_returns.numpy()
                )

        # Periodic evaluation
        iteration += 1
        if iteration % args.evaluate_each == 0:
            returns = [evaluate_episode() for _ in range(args.evaluate_for)]
            mean = np.mean(returns)
            print(f"Evaluation mean: {mean}.")

            if mean >= best_return:
                best_return = mean
                agent.save_models(f"{args.model_path}_{round(mean)}")
                agent.save_args(f"{args.model_path}_{round(mean)}.json", args)
                print(f"New best performing model saved. Score: {best_return}, path: {args.model_path}_{round(mean)}")
            if mean >= 500:
                print(f"Target reached.")
                break

    agent.save_models(f"{args.model_path}_finall")
    agent.save_args(f"{args.model_path}__finall" + ".json", args)

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)
