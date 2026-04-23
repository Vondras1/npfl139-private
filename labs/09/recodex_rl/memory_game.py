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
parser.add_argument("--batch_size", default=32, type=int, help="Number of episodes to train on.")
parser.add_argument("--evaluate_each", default=1280, type=int, help="Evaluate each number of episodes.") # SP: Length of an epoch
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate for number of episodes.") # SP: Number of epochs
parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 3/2*`cards`")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--model_path", default="classic", type=str, help="Model path")
parser.add_argument("--load_model_path_4", default="classic_4", type=str, help="Model path of pretrained model we want to load.")
parser.add_argument("--load_model_path_8", default="classic_8", type=str, help="Model path of pretrained model we want to load.")
parser.add_argument("--load_model_path_16", default="classic_16", type=str, help="Model path of pretrained model we want to load.")
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
                # TODO: Create suitable layers.

                # This layer generates key (query) for memory read from the input observation, which is a (card, symbol) pair.
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
                single = False
                if memory.dim() == 2:
                    memory = memory.unsqueeze(0)      # [1, M, D]
                    single = True
                if observation.dim() == 1:
                    observation = observation.unsqueeze(0)  # [1, 2]

                # Encode the input observation, which is a (card, observation) pair,
                # by representing each element as one-hot and concatenating them, resulting
                # in a vector of length `sum(env.observation_space.nvec)`.
                encoded_input = torch.cat([torch.nn.functional.one_hot(torch.relu(observation[:, i]), dim).float()
                                           for i, dim in enumerate(env.observation_space.nvec)], dim=-1)

                # TODO: Generate a read key for memory read from the encoded input, by using
                # a ReLU hidden layer of size `args.hidden_layer` followed by a dense layer
                # with `args.memory_cell_size` units and `tanh` activation (to keep the memory
                # content in limited range).
                key = self.key_generator(encoded_input)
                key = key.unsqueeze(1)  # [B, 1, D]

                # TODO: Read the memory using the generated read key. Notably, compute cosine
                # similarity of the key and every memory row, apply softmax to generate
                # a weight distribution over the rows, and finally take a weighted average of
                # the memory rows.
                key_norm = torch.norm(key, dim=-1)       # [B, 1]
                memory_norm = torch.norm(memory, dim=-1) # [B, M]
                similarity = torch.sum(key * memory, dim=-1) / (key_norm * memory_norm + 1e-8)
                weights = torch.softmax(similarity, dim=-1) # apply softmax to generate a weight distribution over the rows
                read_memory_value = torch.sum(weights.unsqueeze(-1) * memory, dim=1) # Take a weighted average of the memory rows

                # TODO: Using concatenated encoded input and the read value, use a ReLU hidden
                # layer of size `args.hidden_layer` followed by a dense layer with
                # `env.action_space.n` units to produce policy logits.
                policy_input = torch.cat([encoded_input, read_memory_value], dim=-1)
                policy_logits = self.policy_generator(policy_input)
                
                # TODO: Perform memory write. For faster convergence, add directly
                # the `encoded_input` to the memory, i.e., prepend it as a first memory row
                # and drop the last memory row to keep memory size constant.
                updated_memory = torch.cat([encoded_input.unsqueeze(1), memory[:, :-1, :]], dim=1)

                # TODO: Return the updated memory and the policy
                return updated_memory, policy_logits

        # Create the agent
        self._model = Model().to(self.device)

        # TODO: Create an optimizer and a loss function.
        self._model = Model().to(self.device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)
        self._loss_function = torch.nn.CrossEntropyLoss(reduction="mean") # Sequences may have different length

    def zero_memory(self):
        # TODO: Return an empty memory. It should be a tensor
        # with shape `[self.args.memory_cells, self.args.memory_cell_size]` on `self.device`.
        return torch.zeros((self.args.memory_cells, self.args.memory_cell_size), device=self.device)

    @npfl139.typed_torch_function(device, torch.int64, torch.int64, torch.int64)
    def _train(self, observations, targets, lengths):
        # TODO: Given a batch of sequences of `observations` (each being a (card, symbol) pair),
        # train the network to predict the required `targets`.
        #
        # Specifically, start with a batch of empty memories, and run the agent
        # sequentially as many times as necessary, using `targets` as gold labels.
        #
        # Note that the sequences can be of different length, so you need to pad them
        # to same length and then somehow indicate the length of the individual episodes
        # (one possibility is to add another parameter to `_train`).
        memory = torch.zeros((observations.shape[0], self.args.memory_cells, self.args.memory_cell_size), device=self.device)
        loss = 0

        for t in range(observations.shape[1]): # observations.shape[1] = max_time
            active = lengths > t
            updated_memory, policy_logits = self._model(memory[active], observations[active, t])
            loss += self._loss_function(policy_logits, targets[active, t])
            memory[active] = updated_memory

        # backprop
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def train(self, episodes):
        # TODO: Given a list of episodes, prepare the arguments
        # of the self._train method, and execute it.

        filtered_episodes = []
        for episode in episodes:
            filtered = [(observation, action) for observation, action in episode if action is not None]
            filtered_episodes.append(filtered)

        lengths = torch.tensor([len(episode) for episode in filtered_episodes], device=self.device)
        max_length = int(lengths.max().item())

        # [batch, time, 2], where 2 = raw observation (card, symbol)
        observations = torch.zeros((len(filtered_episodes), max_length, 2), device=self.device, dtype=torch.int64)
        targets = torch.zeros((len(filtered_episodes), max_length), device=self.device, dtype=torch.int64)

        for i, episode in enumerate(filtered_episodes):
            for t, (observation, action) in enumerate(episode):
                observations[i, t] = torch.tensor(observation, device=self.device, dtype=torch.int64)
                targets[i, t] = action

        self._train(observations, targets, lengths)


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
    elif args.cards == 8 and (args.load_pretrained or args.recodex):
        agent.load_models(args.load_model_path_8)
    elif args.cards == 16 and (args.load_pretrained or args.recodex):
        agent.load_models(args.load_model_path_16)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        observation, memory = env.reset(start_evaluation=start_evaluation, logging=logging)[0], agent.zero_memory()
        rewards, done = 0, False
        while not done:
            # TODO: Find out which action to use.
            memory, actions_distribution = agent.predict(memory, torch.tensor(observation, device=agent.device).unsqueeze(0))
            # action = torch.argmax(actions_distribution, dim=-1).item()
            action = int(np.argmax(actions_distribution))
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Training
    training = True
    while training:
        # Generate required number of episodes
        for _ in range(args.evaluate_each // args.batch_size):
            episodes = []
            for _ in range(args.batch_size):
                episodes.append(env.expert_episode())

            # Train the agent
            agent.train(episodes)

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        mean_return = np.mean(returns)
        print(f"Evaluation after {env.episode} episodes: mean return {mean_return:.2f}")

        if mean_return >= 1:
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
