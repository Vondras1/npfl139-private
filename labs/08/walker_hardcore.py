#!/usr/bin/env python3
# Team:
# 1ac5d633-f96f-42a3-846d-31bcb01d041f
# e0cfa255-0259-11eb-9574-ea7484399335
# 9fafb47f-e1c5-4d7c-8ce5-8a6f5bdcd751

import argparse
import collections
import copy
import json
import os

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
# parser.add_argument("--env", default="BipedalWalker-v3", type=str, help="Environment.")
parser.add_argument("--env", default="BipedalWalkerHardcore-v3", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--envs", default=8, type=int, help="Environments.")
parser.add_argument("--evaluate_each", default=5000, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for_shorter", default=20, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--evaluate_for_longer", default=40, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="walker_hard_models/walker", type=str, help="Model path")
parser.add_argument("--load_model_path", default="walker_hard_models/walker_-60", type=str, help="Model path of pretrained model we want to load.")
parser.add_argument("--replay_buffer_size", default=1_000_000, type=int, help="Replay buffer size")
parser.add_argument("--target_entropy", default=-1, type=float, help="Target entropy per action component.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--target_return", default=150, type=float, help="Target return.")
parser.add_argument("--load_pretrained_models", default=False, action="store_true", help="Load pretrained models.")

class Agent:
    # Use GPU if available.
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create an actor.
        class Actor(torch.nn.Module):
            def __init__(self, hidden_layer_size: int):
                super().__init__()
                # TODO: Create
                # - two hidden layers with `hidden_layer_size` and ReLU activation
                # - a layer for generating means with `env.action_space.shape[0]` units and no activation
                # - a layer for generating sds with `env.action_space.shape[0]` units and `torch.exp` activation
                self.first_layer = torch.nn.Linear(env.observation_space.shape[0], hidden_layer_size)
                self.second_layer = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
                self.generate_means = torch.nn.Linear(hidden_layer_size, env.action_space.shape[0])
                self.generate_sds = torch.nn.Linear(hidden_layer_size, env.action_space.shape[0])

                self.relu = torch.nn.ReLU()

                # Then, create a variable representing a logarithm of alpha, using for example the following:
                self._log_alpha = torch.nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))

                # Finally, create two tensors representing the action scale and offset.
                self.register_buffer("action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2))
                self.register_buffer("action_offset", torch.tensor((env.action_space.high + env.action_space.low) / 2))

            def forward(self, inputs: torch.Tensor, sample: bool):
                # TODO: Perform the actor computation
                # - First, pass the inputs through the first hidden layer
                #   and then through the second hidden layer.
                x = self.relu(self.first_layer(inputs))
                x = self.relu(self.second_layer(x))

                # - From these hidden states, compute
                #   - `mus` (the means),
                #   - `sds` (the standard deviations).
                mus = self.generate_means(x)
                sds = torch.exp(self.generate_sds(x))

                # - Then, create the action distribution using `torch.distributions.Normal`
                #   with the `mus` and `sds`.
                action_distribution = torch.distributions.Normal(mus, sds)

                # - We then bijectively modify the distribution so that the actions are
                #   in the given range. Luckily, `torch.distributions.transforms` offers
                #   a class `torch.distributions.TransformedDistribution` than can transform
                #   a distribution by a given transformation. We need to use
                #   - `torch.distributions.transforms.TanhTransform()`
                #     to squash the actions to [-1, 1] range, and then
                #   - `torch.distributions.transforms.AffineTransform(self.action_offset, self.action_scale)`
                #     to scale the action ranges to [low, high].
                #   - To compose these transformations, use
                #     `torch.distributions.transforms.ComposeTransform([t1, t2], cache_size=1)`
                #     with `cache_size=1` parameter for numerical stability.
                #   Note that the `ComposeTransform` can be created already in the constructor
                #   for better performance.
                #   In case you wanted to do this manually, sample from a normal distribution, pass the samples
                #   through the `tanh` and suitable scaling, and then compute the log-prob by using `log_prob`
                #   from the normal distribution and manually accounting for the `tanh` as shown in the slides.
                #   However, the formula from the slides is not numerically stable, for a better variant see
                #   https://github.com/tensorflow/probability/blob/ef1f64a434/tensorflow_probability/python/bijectors/tanh.py#L70-L81
                self.compose_transform = torch.distributions.transforms.ComposeTransform(
                    [
                        torch.distributions.transforms.TanhTransform(),
                        torch.distributions.transforms.AffineTransform(self.action_offset, self.action_scale)
                    ], cache_size=1
                )
                action_distribution = torch.distributions.TransformedDistribution(action_distribution, self.compose_transform)

                # - Sample the actions by a `rsample()` call (`sample()` is not differentiable).
                if sample:
                    actions = action_distribution.rsample()
                else:
                    actions = torch.tanh(mus) * self.action_scale + self.action_offset
                
                # - Then, compute the log-probabilities of the sampled actions by using `log_prob()`
                #   call. An action is actually a vector, so to be precise, compute for every batch
                #   element a scalar, an average of the log-probabilities of individual action components.
                # log_prob = action_distribution.log_prob(actions)
                log_prob = action_distribution.log_prob(actions).mean(dim=-1, keepdim=True)

                # - Finally, compute `alpha` as exponentiation of `self._log_alpha`.
                alpha = torch.exp(self._log_alpha)

                # - Return actions, log_prob, and alpha.
                return actions, log_prob, alpha
            
                #
                # Do not forget to support computation without sampling (`sample==False`). You
                # can return for example `torch.tanh(mus) * self.action_scale + self.action_offset`,
                # or you can use for example `sds=1e-7`.

        # Instantiate the actor as `self._actor`.
        self._actor = Actor(args.hidden_layer_size).to(self.device)

        # TODO: Create a critic, which
        # - takes observations and actions as inputs,
        # - concatenates them,
        # - passes the result through two dense layers with `args.hidden_layer_size` units
        #   and ReLU activation,
        # - finally, using a last dense layer produces a single output with no activation
        # This critic needs to be cloned (for example using `copy.deepcopy`) so that
        # two critics and two target critics are created. Note that the critics should be
        # different with respect to each other, but the target critics should be the same
        # as their corresponding original critics.
        class Critic(torch.nn.Module):
            def __init__(self, hidden_layer_size: int):
                super().__init__()
                self.first_layer = torch.nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], hidden_layer_size)
                self.second_layer = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
                self.output_layer = torch.nn.Linear(hidden_layer_size, 1)
                self.relu = torch.nn.ReLU()
            
            def forward(self, inputs: torch.Tensor, actions: torch.Tensor):
                x = torch.cat([inputs, actions], dim=-1)
                x = self.relu(self.first_layer(x))
                x = self.relu(self.second_layer(x))
                Q = self.output_layer(x)
                return Q
        
        # Instantiate the critics as `self._critic{i}`.
        self._critic1 = Critic(args.hidden_layer_size).to(self.device)
        self._critic2 = Critic(args.hidden_layer_size).to(self.device)

        # Initialize target critics by copying the current ones
        self._target_critic1 = copy.deepcopy(self._critic1)
        self._target_critic2 = copy.deepcopy(self._critic2)

        # Save args:
        self.args = args

        self._target_entropy = args.target_entropy * env.action_space.shape[0]

        # TODO: Define an optimizer. Using `torch.optim.Adam` optimizer with
        # the given `args.learning_rate` is a good default.

        # self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=args.learning_rate)
        # self._critic_optimizer = torch.optim.Adam(list(self._critic1.parameters()) + list(self._critic2.parameters()), lr=args.learning_rate)
        # self._alpha_optimizer = torch.optim.Adam(self._actor.parameters(), lr=args.learning_rate)
        # self._optimizer = torch.optim.Adam(list(self._actor.parameters()) + list(self._critic1.parameters()) + list(self._critic2.parameters()), lr=args.learning_rate)

        self._actor_optimizer = torch.optim.Adam(
            [p for n, p in self._actor.named_parameters() if n != "_log_alpha"],
            lr=args.learning_rate
        )

        self._critic_optimizer = torch.optim.Adam(
            list(self._critic1.parameters()) + list(self._critic2.parameters()),
            lr=args.learning_rate
        )

        self._alpha_optimizer = torch.optim.Adam(
            [self._actor._log_alpha],
            lr=args.learning_rate
        )

        # Create MSE loss.
        self._mse_loss = torch.nn.MSELoss()

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Separately train:

        # Actor update
        # - the actor, by using two objectives:
        #   - the objective for the actor itself; in this objective, `alpha.detach()`
        #     should be used (for the `alpha` returned by the actor) to avoid optimizing `alpha`,
        new_actions, log_prob, alpha = self._actor(states, sample=True)
        q1 = self._critic1(states, new_actions)
        q2 = self._critic2(states, new_actions)

        actor_loss = (alpha.detach() * log_prob - torch.minimum(q1,q2)).mean()
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        # Alpha update
        #   - the objective for `alpha`, where `log_prob.detach()` should be used
        #     to avoid computing gradient for other variables than `alpha`.
        #     Use `args.target_entropy` as the target entropy (the default of -1 per action
        #     component is fine and does not need to be tuned for the agent to train).
        _, log_prob_alpha, alpha = self._actor(states, sample=True) # Use updated actor
        alpha_loss = (-alpha * log_prob_alpha.detach() - alpha * self._target_entropy).mean()
        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self._alpha_optimizer.step()

        # Critics update
        # - the critics using MSE loss.
        q1 = self._critic1(states, actions).squeeze(-1)
        q2 = self._critic2(states, actions).squeeze(-1)

        critic_loss = self._mse_loss(q1, returns) + self._mse_loss(q2, returns)
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        #
        # Finally, update the two target critic networks exponential moving
        # average with weight `args.target_tau`, using for example the provided
        #   npfl139.update_params_by_ema(target: torch.nn.Module, source: torch.nn.Module, tau: float)
        npfl139.update_params_by_ema(self._target_critic1, self._critic1, self.args.target_tau)
        npfl139.update_params_by_ema(self._target_critic2, self._critic2, self.args.target_tau)

    # Predict actions without sampling.
    @npfl139.typed_torch_function(device, torch.float32)
    def predict_mean_actions(self, states: torch.Tensor) -> np.ndarray:
        # Return predicted actions.
        # states /= self._observation_divisor or 1
        with torch.no_grad():
            return self._actor(states, sample=False)[0]

    # Predict actions with sampling.
    @npfl139.typed_torch_function(device, torch.float32)
    def predict_sampled_actions(self, states: torch.Tensor) -> np.ndarray:
        # Return sampled actions from the predicted distribution
        # states /= self._observation_divisor or 1
        with torch.no_grad():
            return self._actor(states, sample=True)[0]

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Produce the predicted returns, which are the minimum of
        #    target_critic(s, a) - alpha * log_prob
        #  considering both target critics and actions sampled from the actor.
        actions, log_prob, alpha = self._actor(states, sample=True)
        q1 = self._target_critic1(states, actions)
        q2 = self._target_critic2(states, actions)
        return torch.minimum(q1, q2) - alpha * log_prob

    # Serialization methods.
    def save_actor(self, path: str) -> None:
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str) -> None:
        self._actor.load_state_dict(torch.load(path, map_location=self.device))
    
    def save_models(self, path: str) -> None:
        torch.save({
            "actor": self._actor.state_dict(),
            "critic1": self._critic1.state_dict(),
            "critic2": self._critic2.state_dict(),
            "target_critic1": self._target_critic1.state_dict(),
            "target_critic2": self._target_critic2.state_dict(),
        }, path)
    
    def load_models(self, path: str) -> None:
        model_blocks = torch.load(path, map_location=self.device)

        self._actor.load_state_dict(model_blocks["actor"])
        self._critic1.load_state_dict(model_blocks["critic1"])
        self._critic2.load_state_dict(model_blocks["critic2"])
        self._target_critic1.load_state_dict(model_blocks["target_critic1"])
        self._target_critic2.load_state_dict(model_blocks["target_critic2"])

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
            # TODO: Predict an action using the greedy policy.
            action = agent.predict_mean_actions(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    best_return = -100
    if args.load_pretrained_models or args.recodex:
        agent.load_models(args.load_model_path)
        score = extract_score_from_model_path(args.load_model_path)
        if score is not None:
            best_return = score
            print(f"Loaded best_return: {best_return}")
    
    # ReCodEx evaluation.
    if args.recodex:
        while True:
            evaluate_episode(start_evaluation=True)

    # Create the asynchronous vector environment for training.
    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC,
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})

    # Replay memory of a specified maximum size.
    replay_buffer = npfl139.ReplayBuffer(args.replay_buffer_size, args.seed)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    state = vector_env.reset(seed=args.seed)[0]
    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # Predict actions by calling `agent.predict_sampled_actions`.
            action = agent.predict_sampled_actions(state)

            next_state, reward, terminated, truncated, _ = vector_env.step(action)
            done = terminated | truncated

            # Remove terminal penalty # TODO
            reward = np.where(done, 0.0, reward)

            replay_buffer.append_batch(Transition(state, action, reward, done, next_state))
            state = next_state

            # Training
            if len(replay_buffer) >= 15 * args.batch_size:
                # Randomly uniformly sample transitions from the replay buffer.
                states, actions, rewards, dones, next_states = replay_buffer.sample(args.batch_size)
                # TODO: Perform the training
                next_values = agent.predict_values(next_states).squeeze(-1)
                returns = rewards + args.gamma * (~dones) * next_values
                agent.train(states, actions, returns)

        # Periodic evaluation
        returns_short = [evaluate_episode() for _ in range(args.evaluate_for_shorter)]
        avg_return_short = np.mean(returns_short)
        print("Average short evaluation return is: ", avg_return_short)

        if avg_return_short >= best_return or avg_return_short >= 80:
            avg_return_long = np.mean([evaluate_episode() for _ in range(args.evaluate_for_longer)])
            print("Average long evaluation return is: ", avg_return_long)
            if avg_return_long >= best_return:
                best_return = avg_return_long
                agent.save_models(f"{args.model_path}_{round(avg_return_long)}")
                agent.save_args(f"{args.model_path}_{round(avg_return_long)}.json", args)
            elif(avg_return_long >= 200 and avg_return_long >= (best_return-20)):
                agent.save_models(f"{args.model_path}_{round(avg_return_long)}")
                agent.save_args(f"{args.model_path}_{round(avg_return_long)}.json", args)
            if avg_return_long >= args.target_return:
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
