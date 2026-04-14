#!/usr/bin/env python3
# Team:
# 1ac5d633-f96f-42a3-846d-31bcb01d041f
# e0cfa255-0259-11eb-9574-ea7484399335
# 9fafb47f-e1c5-4d7c-8ce5-8a6f5bdcd751

import argparse

import gymnasium as gym
import numpy as np
import torch
import json

import npfl139
npfl139.require_version("2526.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=32, type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=20, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--tiles", default=10, type=int, help="Tiles to use.")
parser.add_argument("--model_path", default="paac_continuous_actor.pt", type=str, help="Path to the actor model.")



class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Analogously to paac, your model should contain two components:
        # - an actor, which predicts distribution over the actions, and
        # - a critic, which predicts the value function.
        #
        # The given states are tile encoded, so they are integer indices of
        # tiles intersecting the state. Therefore, you should convert them
        # to dense encoding (one-hot-like, with `args.tiles` ones); or you can
        # even use the `torch.nn.EmbeddingBag` layer.
        #
        # The actor computes `mus` and `sds`, each of shape `[batch_size, actions]`.
        # Compute each independently using states as input, adding a fully connected
        # layer with `args.hidden_layer_size` units and a ReLU activation. Then:
        # - For `mus`, add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required range, you should apply
        #   properly scaled `torch.tanh` activation.
        # - For `sds`, add a fully connected layer with `actions` outputs
        #   and `torch.exp` or `torch.nn.functional.softplus` activation.
        #
        # The critic should be a usual one, passing states through one hidden
        # layer with `args.hidden_layer_size` ReLU units and then predicting
        # the value function.     
        super().__init__()

        self._input_weights = env.observation_space.nvec[-1]
        self._action_dim = env.action_space.shape[0]
        # print(self._input_weights)  # Box(-1.0, 1.0, (1,), float32)

        # actor - policy network
        # mus
        self._actor_mus = torch.nn.Sequential(
            torch.nn.Linear(self._input_weights, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, self._action_dim),
            torch.nn.Tanh()
                ).to(self.device)

        # sds
        self._actor_sds = torch.nn.Sequential(
            torch.nn.Linear(self._input_weights, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, self._action_dim),
            torch.nn.Softplus()
                ).to(self.device) 
             

        # critic - value network
        self._critic = torch.nn.Sequential(
            torch.nn.Linear(self._input_weights, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1),
                ).to(self.device)

        self._actor_optimizer = torch.optim.Adam(
            list(self._actor_mus.parameters()) + list(self._actor_sds.parameters()),
            lr=args.learning_rate)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=args.learning_rate)
        self._entropy_regularization = args.entropy_regularization

        return
    

    def _encode_states(self, states: torch.Tensor) -> torch.Tensor:
        states = states.long()  # converts to long ints
        if states.ndim == 1:
            states = states.unsqueeze(0)  # unsqueezes to batch dim

        # one-hot encoding
        x = torch.zeros(states.shape[0], self._input_weights, device=states.device)   
        x.scatter_(1, states, 1.0)                                          
        return x

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.int64, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Run the model on given `states` and compute `sds`, `mus` and predicted values.
        # Then create `action_distribution` using `torch.distributions.Normal` class and
        # the computed `mus` and `sds`.
        #
        # TODO: Train the actor using the sum of the following two losses:
        # - REINFORCE loss, i.e., the negative log likelihood of the `actions` in the
        #   `action_distribution` (using the `log_prob` method). You then need to sum
        #   the log probabilities of the action components in a single batch example.
        #   Finally, multiply the resulting vector by `(returns - baseline)`
        #   and compute its mean. Be sure to let the gradient flow only where it should.
        # - negative value of the distribution entropy (use `entropy` method of
        #   the `action_distribution`) weighted by `args.entropy_regularization`.
        #
        # Train the critic using mean square error of the `returns` and predicted values.

        # setting to trains
        self._actor_mus.train()
        self._actor_sds.train()
        self._critic.train()

        # encoding states to one-hot
        x = self._encode_states(states)

        # computing action distr and values
        mus = self._actor_mus(x)
        sds = self._actor_sds(x)   + 1e-6
        sds = torch.clamp(sds, min=1e-6, max=1e6)
        values = self._critic(x).squeeze(-1)

        action_distribution = torch.distributions.Normal(mus, sds)
        log_probs = action_distribution.log_prob(actions).sum(dim=-1)
        entropy = action_distribution.entropy().sum(dim=-1).mean()

        ## losses
        advantages = returns - values.detach()
        loss_actor = -(advantages * log_probs).mean() - self._entropy_regularization * entropy
        loss_critic = torch.nn.functional.mse_loss(values, returns)

        # grad steps 
        self._actor_optimizer.zero_grad()
        loss_actor.backward()
        self._actor_optimizer.step()

        self._critic_optimizer.zero_grad()
        loss_critic.backward()
        self._critic_optimizer.step()  
        
        return

    @npfl139.typed_torch_function(device, torch.int64)
    def predict_actions(self, states: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return predicted action distributions (mus and sds).
        self._actor_mus.eval()
        self._actor_sds.eval()

        with torch.no_grad():
            x = self._encode_states(states)
            mus = self._actor_mus(x)
            sds = self._actor_sds(x)  + 1e-6
            sds = torch.clamp(sds, min=1e-6, max=1e6)          

        return mus, sds

    @npfl139.typed_torch_function(device, torch.int64)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        self._critic.eval()

        with torch.no_grad():
            x = self._encode_states(states)
            values = self._critic(x).squeeze(1) # [batch_size, 1] -> [batch_size]

        return values

    # Serialization methods.
    def save_actor(self, path: str) -> None:
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str) -> None:
        self._actor.load_state_dict(torch.load(path, map_location=self.device))

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
    agent = Agent(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # TODO: Predict an action using the greedy policy.
            mus, _ = agent.predict_actions(state)   # 
            action = mus[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment, using the SAME_STEP autoreset mode.
    vector_env = gym.make_vec("MountainCarContinuous-v0", args.envs, gym.VectorizeMode.ASYNC,
                              wrappers=[lambda env: npfl139.DiscreteMountainCarWrapper(env, tiles=args.tiles)],
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    states = vector_env.reset(seed=args.seed)[0]

    training = True
    returns_best = -np.inf

    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Predict action distribution using `agent.predict_actions`
            # and then sample it using for example `np.random.normal`. Do not
            # forget to clip the actions to the `env.action_space.{low,high}`
            # range, for example using `np.clip`.
            mus, sds = agent.predict_actions(states)
            actions = np.random.normal(mus, sds)
            actions = np.clip(actions, env.action_space.low, env.action_space.high)

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = terminated | truncated
            next_values = agent.predict_values(next_states)

            # TODO(paac): Compute estimates of returns by one-step bootstrapping
            returns = rewards + args.gamma * next_values * (1 - dones.astype(np.float32))


            # TODO(paac): Train agent using current states, chosen actions and estimated returns.
            agent.train(states, actions, returns)

            states = next_states

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]

        if np.mean(returns) > returns_best:
            returns_best = np.mean(returns)
            # agent.save_actor(args.model_path)
            # agent.save_args(args.model_path + ".json", args)
            print("returns_best:", returns_best, "\t") 

        if np.mean(returns) >= 93:        
            training = False


    # # Save the agent
    # agent.save_actor(args.model_path)
    # agent.save_args(args.model_path + ".json", args)


    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=main_args.tiles),
        main_args.seed, main_args.render_each)

    main(main_env, main_args)
