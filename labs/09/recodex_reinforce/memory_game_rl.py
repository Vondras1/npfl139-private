#!/usr/bin/env python3
import argparse
import json

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.9")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cards", default=8, type=int, help="Number of cards in the memory game.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=64, type=int, help="Number of episodes to train on.")
parser.add_argument("--gradient_clipping", default=1.0, type=float, help="Gradient clipping.")
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
parser.add_argument("--evaluate_each", default=5000, type=int, help="Evaluate each number of episodes.") # SP: Length of an epoch
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate for number of episodes.") # SP: Number of epochs
parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 3/2*`cards`")
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--model_path", default="memory_models/reinforce", type=str, help="Model path")
parser.add_argument("--load_model_path_4", default="reinforce_4", type=str, help="Model path of pretrained model we want to load.")
parser.add_argument("--load_model_path_6", default="reinforce_6", type=str, help="Model path of pretrained model we want to load.")
parser.add_argument("--load_model_path_8", default="reinforce_8", type=str, help="Model path of pretrained model we want to load.")
parser.add_argument("--load_pretrained", default=False, action="store_true", help="Load pretrained models.")

class Agent:
    device = torch.device("cpu")
    # Use the following line instead to use GPU if available.
    # device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args
        self.env = env

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # TODO(memory_game): Create suitable layers.
                self.key_generator = torch.nn.Sequential(
                    torch.nn.Linear(args.memory_cell_size, args.hidden_layer),
                    torch.nn.ReLU(),
                    torch.nn.Linear(args.hidden_layer, args.memory_cell_size),
                    torch.nn.Tanh()
                )

                self.policy_generator = torch.nn.Sequential(
                    torch.nn.Linear(args.memory_cell_size + sum(env.observation_space.nvec), args.hidden_layer),
                    torch.nn.ReLU(),
                    torch.nn.Linear(args.hidden_layer, env.action_space.n)
                )

            def forward(self, memory, observation):
                # Encode the input observation, which is a (card, observation) pair,
                # by representing each element as one-hot and concatenating them, resulting
                # in a vector of length `sum(env.observation_space.nvec)`.
                encoded_input = torch.cat([torch.nn.functional.one_hot(torch.relu(observation[:, i]), dim).float()
                                           for i, dim in enumerate(env.observation_space.nvec)], dim=-1)

                # TODO(memory_game): Generate a read key for memory read from the encoded input, by using
                # a ReLU hidden layer of size `args.hidden_layer` followed by a dense layer
                # with `args.memory_cell_size` units and `tanh` activation (to keep the memory
                # content in limited range).
                key = self.key_generator(encoded_input)
                key = key.unsqueeze(1)  # [B, 1, D]

                # TODO(memory_game): Read the memory using the generated read key. Notably, compute cosine
                # similarity of the key and every memory row, apply softmax to generate
                # a weight distribution over the rows, and finally take a weighted average of
                # the memory rows.
                key_norm = torch.norm(key, dim=-1)       # [B, 1]
                memory_norm = torch.norm(memory, dim=-1) # [B, M]
                similarity = torch.sum(key * memory, dim=-1) / (key_norm * memory_norm + 1e-8)
                weights = torch.softmax(similarity, dim=-1) # apply softmax to generate a weight distribution over the rows
                read_memory_value = torch.sum(weights.unsqueeze(-1) * memory, dim=1) # Take a weighted average of the memory rows

                # TODO(memory_game): Using concatenated encoded input and the read value, use a ReLU hidden
                # layer of size `args.hidden_layer` followed by a dense layer with
                # `env.action_space.n` units to produce policy logits.
                policy_input = torch.cat([encoded_input, read_memory_value], dim=-1)
                policy_logits = self.policy_generator(policy_input)

                # TODO(memory_game): Perform memory write. For faster convergence, add directly
                # the `encoded_input` to the memory, i.e., prepend it as a first memory row
                # and drop the last memory row to keep memory size constant.
                updated_memory = torch.cat([encoded_input.unsqueeze(1), memory[:, :-1, :]], dim=1)

                # TODO(memory_game): Return the updated memory and the policy
                return updated_memory, policy_logits

        # Create the agent
        self._model = Model().to(self.device)

        # TODO(memory_game): Create an optimizer and a loss function.
        self._optimizer = torch.optim.Adam(params=self._model.parameters(), lr=args.learning_rate)
        self._loss = torch.nn.CrossEntropyLoss(reduction="none")

        # baseline
        self._baseline = None

    def zero_memory(self):
        # TODO(memory_game): Return an empty memory. It should be a tensor
        # with shape `[self.args.memory_cells, self.args.memory_cell_size]` on `self.device`.
        return torch.zeros((1, self.args.memory_cells, self.args.memory_cell_size), device=self.device)

    @npfl139.typed_torch_function(device, torch.int64, torch.int64, torch.float32, torch.int64)
    def _train(self, observations, actions, returns, lengths):
        # TODO: Train the network given a batch of sequences of `observations`
        # (each being a (card, symbol) pair), sampled `actions` and observed `returns`.
        # Specifically, start with a batch of empty memories, and run the agent
        # sequentially as many times as necessary, using `actions` as actions.
        #
        # Use the REINFORCE algorithm, optionally with a baseline. Note that
        # I use a baseline, but not a baseline computed by a neural network;
        # instead, for every time step, I track exponential moving average of
        # observed returns, with momentum 0.01. Furthermore, I use entropy regularization
        # with coefficient `args.entropy_regularization`.
        #
        # Note that the sequences can be of different length, so you need to pad them
        # to same length and then somehow indicate the length of the individual episodes
        # (one possibility is to add another parameter to `_train`).
        self._model.train()

        # Constants
        batch_size, max_length = observations.shape[:2]
        loss = 0
        ema_momentum = 0.01
        valid_steps = 0

        # Initialize memory
        memory = torch.zeros((batch_size, self.args.memory_cells, self.args.memory_cell_size), device=self.device)
        
        # Initialize baseline
        if self._baseline is None or len(self._baseline) < max_length:
            self._baseline = torch.zeros(max_length, device=self.device)
        
        for t in range(max_length):
            active = lengths > t

            updated_memory, policy_logits = self._model(memory[active], observations[active, t])
            ce_loss = self._loss(policy_logits, actions[active, t]) # CrossEntropyLoss(logits, action) = -log pi(action | state)

            # EMA at timestep t
            mean_return_t = returns[active, t].mean().detach()
            if self._baseline[t] == 0:
                self._baseline[t] = mean_return_t
            else:
                self._baseline[t] = (1 - ema_momentum) * self._baseline[t] + ema_momentum * mean_return_t

            advantages = (returns[active, t] - self._baseline[t]).detach()  

            policy_loss = (advantages * ce_loss).mean()

            distribution = torch.distributions.Categorical(logits=policy_logits)
            entropy = distribution.entropy()

            loss = loss + (policy_loss - self.args.entropy_regularization * entropy).mean()

            memory[active] = updated_memory
            valid_steps += 1

        loss = loss / valid_steps

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.args.gradient_clipping)
        self._optimizer.step()

    def train(self, episodes):
        # TODO: Given a list of episodes, prepare the arguments
        # of the self._train method, and execute it.
        lengths = torch.tensor([len(episode) for episode in episodes], device=self.device)
        max_length = int(lengths.max().item())

        # [batch, time, 2], where 2 = raw observation (card, symbol)
        observations = torch.zeros((len(episodes), max_length, 2), device=self.device, dtype=torch.int64)
        actions = torch.zeros((len(episodes), max_length), device=self.device, dtype=torch.int64)
        returns = torch.zeros((len(episodes), max_length), device=self.device, dtype=torch.float32)

        for i, episode in enumerate(episodes):
            for t, (observation, action, reward, _return) in enumerate(episode):
                observations[i, t] = torch.tensor(observation, device=self.device, dtype=torch.int64)
                actions[i, t] = action
                returns[i, t] = _return
        
        self._train(observations, actions, returns, lengths)
 

    @npfl139.typed_torch_function(device, torch.float32, torch.int64)
    def predict(self, memory, observation):
        self._model.eval()
        with torch.no_grad():
            memory, logits = self._model(memory, observation)
            return memory, torch.softmax(logits, dim=-1)

    def save_models(self, path: str) -> None:
        torch.save(self._model.state_dict(), path + "_model.pt")
    
    def load_models(self, path: str) -> None:
        self._model.load_state_dict(torch.load(path + "_model.pt", map_location=self.device))

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

    # Post-process arguments to default values if not overridden on the command line.
    if args.hidden_layer is None:
        args.hidden_layer = 8 * args.cards
    if args.memory_cells is None:
        args.memory_cells = 2 * args.cards
    if args.memory_cell_size is None:
        args.memory_cell_size = 3 * args.cards // 2
    assert sum(env.observation_space.nvec) == args.memory_cell_size

    # Construct the agent.
    agent = Agent(env, args)

    if args.cards == 4 and (args.load_pretrained or args.recodex):
        agent.load_models(args.load_model_path_4)
    elif args.cards == 6 and (args.load_pretrained or args.recodex):
        agent.load_models(args.load_model_path_6)
    elif args.cards == 8 and (args.load_pretrained or args.recodex):
        agent.load_models(args.load_model_path_8)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        observation, memory = env.reset(start_evaluation=start_evaluation, logging=logging)[0], agent.zero_memory()
        rewards, done = 0, False
        while not done:
            # TODO(memory_game): Find out which action to use.
            memory, actions_distribution = agent.predict(memory, torch.tensor(observation, device=agent.device).unsqueeze(0))
            action = int(np.argmax(actions_distribution))
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Training
    training = not args.recodex
    while training:
        # Generate required number of episodes
        for _ in range(args.evaluate_each // args.batch_size):
            episodes = []
            for _ in range(args.batch_size):
                observation, memory, episode, done = env.reset()[0], agent.zero_memory(), [], False
                while not done:
                    # TODO: Choose an action according to the generated distribution.
                    memory, policy_probs = agent.predict(memory, torch.tensor(observation, device=agent.device).unsqueeze(0))

                    policy_probs = policy_probs.squeeze()
                    action = np.random.choice(len(policy_probs), p = policy_probs)

                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode.append([observation, action, reward])
                    observation = next_observation

                # TODO: In the `episode`, compute returns from the rewards.
                G = 0
                returns = []

                for _, _, r in reversed(episode):
                    G = r + G
                    returns.insert(0, G)

                episode = [[observation, action, reward, _return] for (observation, action, reward), _return in zip(episode, returns)]
                episodes.append(episode)

            # Train the agent
            agent.train(episodes)

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        mean_return = np.mean(returns)
        print(f"Evaluation after {env.episode} episodes: mean return {mean_return:.2f}")

        if mean_return >= 0.5:
            print(f"Target reached.")
            suffix = f"{mean_return:.2f}"
            agent.save_models(f"{args.model_path}_{args.cards}")
            agent.save_args(f"{args.model_path}_{args.cards}.json", args)
            break

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("npfl139/MemoryGame-v0", cards=main_args.cards), main_args.seed, main_args.render_each)

    main(main_env, main_args)
