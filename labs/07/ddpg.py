#!/usr/bin/env python3
import argparse
import collections
import copy

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="InvertedDoublePendulum-v5", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate.")
parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Replay buffer size")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.") # 0.001 according to the paper

class Actor(torch.nn.Module):
    def __init__(self, state_dim:int, action_dim:int, action_min:torch.Tensor, action_max:torch.Tensor, args: argparse.Namespace):
        super().__init__()
        self.scale = (action_max - action_min) / 2
        self.bias = (action_max + action_min) / 2

        self.input_linear = torch.nn.Linear(state_dim, args.hidden_layer_size)
        self.relu1 = torch.nn.ReLU()
        self.middle_linear = torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size)
        self.relu2 = torch.nn.ReLU()
        self.output_linear = torch.nn.Linear(args.hidden_layer_size, action_dim)
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x):
        x = self.relu1(self.input_linear(x))
        x = self.relu2(self.middle_linear(x))
        x = self.tanh(self.output_linear(x))
        actions = self.scale * x + self.bias
        return actions

class Critic(torch.nn.Module):
    def __init__(self, state_dim:int, action_dim:int, args: argparse.Namespace):
        super().__init__()
        self.input_linear = torch.nn.Linear(state_dim+action_dim, args.hidden_layer_size)
        self.relu1 = torch.nn.ReLU()
        self.middle_linear = torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size)
        self.relu2 = torch.nn.ReLU()
        self.output_linear = torch.nn.Linear(args.hidden_layer_size, 1)
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = self.relu1(self.input_linear(x))
        x = self.relu2(self.middle_linear(x))
        Q = self.output_linear(x)
        return Q

class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create:
        # - An actor, which starts with states and returns actions.
        #   Usually, one or two hidden layers are employed. As in the
        #   paac_continuous, to keep the actions in the required range, you
        #   should apply properly scaled `torch.tanh` activation.
        #
        # - A target actor as the copy of the actor using `copy.deepcopy`.
        #
        # - A critic, starting with given states and actions, producing predicted
        #   returns. The states and actions are usually concatenated and fed through
        #   two more hidden layers, before computing the returns with the last output layer.
        #
        # - A target critic as the copy of the critic using `copy.deepcopy`.

        # Save args:
        self.args = args

        # Environment info:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_min = torch.tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        action_max = torch.tensor(env.action_space.high, dtype=torch.float32, device=self.device)

        # An actor:
        self.actor = Actor(state_dim, action_dim, action_min, action_max, args).to(self.device)

        # A target actor:
        self.target_actor = copy.deepcopy(self.actor)

        # A critic:
        self.critic = Critic(state_dim, action_dim, args).to(self.device)

        # A target critic:
        self.target_critic = copy.deepcopy(self.critic)

        # Initialize losses
        self.critic_loss = torch.nn.MSELoss()
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.learning_rate)


    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Separately train:
        # - the actor using the DPG loss,
        # - the critic using MSE loss.
        #
        # Furthermore, update the target actor and critic networks by exponential moving average
        # with momentum `args.target_tau`. An implementation for EMA update is provided as
        #   npfl139.update_params_by_ema(target: torch.nn.Module, source: torch.nn.Module, tau: float)

        # Update critic:
        loss_c = self.critic_loss(returns, self.critic(states, actions).squeeze(-1))
        self.critic_optimizer.zero_grad() # zero gradients for every batch
        loss_c.backward() # gradients
        self.critic_optimizer.step() # Adjust learning weights

        # Update actor:
        loss_a = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        loss_a.backward()
        self.actor_optimizer.step()

        # Update target actor and critic:
        npfl139.update_params_by_ema(self.target_actor, self.actor, self.args.target_tau)
        npfl139.update_params_by_ema(self.target_critic, self.critic, self.args.target_tau)

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted actions by the actor.
        return self.actor(states)

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted returns -- predict actions by the target actor
        # and evaluate them using the target critic.
        actions = self.target_actor(states)
        returns = self.target_critic(states, actions)
        return returns.squeeze(-1)


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    agent = Agent(env, args)

    # Replay memory of a specified maximum size.
    replay_buffer = npfl139.ReplayBuffer(args.replay_buffer_size, args.seed)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    if args.env == "Pendulum-v1":
        target_return = -175
    elif args.env == "InvertedDoublePendulum-v5":
        target_return = 9100
    else:
        target_return = None

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # TODO: Predict an action by calling `agent.predict_actions`.
            action = agent.predict_actions(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset()[0], False
            noise.reset()
            while not done:
                # TODO: Predict actions by calling `agent.predict_actions`
                # and adding the Ornstein-Uhlenbeck noise. As in paac_continuous,
                # clip the actions to the `env.action_space.{low,high}` range.
                action = agent.predict_actions(state) + noise.sample()
                action = np.clip(action, env.action_space.low, env.action_space.high)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if len(replay_buffer) < 4 * args.batch_size:
                    continue
                states, actions, rewards, dones, next_states = replay_buffer.sample(args.batch_size)
                # TODO: Perform the training
                returns = rewards + args.gamma * agent.predict_values(next_states) * (~dones)
                agent.train(states, actions, returns)

        # Periodic evaluation
        eval_returns = [evaluate_episode(logging=False) for _ in range(args.evaluate_for)]
        print(f"Evaluation after episode {env.episode}: {np.mean(eval_returns):.2f}")
        if np.mean(eval_returns) >= target_return or target_return is None: 
            break

    # Final evaluation
    print("Evaluation started")
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)
