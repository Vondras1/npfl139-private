#!/usr/bin/env python3
import argparse
import collections
import math

from pathlib import Path
import json

import numpy as np
import torch

# Speed up convolutions on fixed-size inputs.
torch.backends.cudnn.benchmark = True

import npfl139
npfl139.require_version("2526.11.2")
from npfl139.board_games import Pisqorky

# c++ import board_game_cpp
import board_game_cpp

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=256, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.15, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=512, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="models/pisqorky_alpha0.15_lr0.001_sim800_sample31.pt", type=str, help="Model path")
parser.add_argument("--num_simulations", default=800, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--replay_buffer_length", default=40000, type=int, help="Replay buffer max length.")
parser.add_argument("--sampling_moves", default=30, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=3, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=10, type=int, help="Update steps in every iteration.")
parser.add_argument("--resume", default=False, action="store_true", help="Continue training from model_path.")

parser.add_argument("--save_dir", default="models", type=str, help="Directory for saved models.")
parser.add_argument("--run_name", default=None, type=str, help="Optional experiment name.")

def prepare_model_path(args: argparse.Namespace) -> Path:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.run_name is None:
        run_name = (
            f"pisqorky"
            f"_alpha{args.alpha}"
            f"_lr{args.learning_rate}"
            f"_sim{args.num_simulations}"
            f"_sample{args.sampling_moves}"
        )
    else:
        run_name = args.run_name

    return save_dir / f"{run_name}.pt"


#########
# Agent #
#########
class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Model(torch.nn.Module):
        def __init__(self, board_shape, num_actions):
            super().__init__()

            filters = 48

            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(3, filters, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )

            self.policy_head = torch.nn.Sequential(
                torch.nn.Conv2d(filters, 2, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(2 * board_shape[1] * board_shape[2], num_actions),
            )

            self.value_head = torch.nn.Sequential(
                torch.nn.Conv2d(filters, 2, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(2 * board_shape[1] * board_shape[2], 1),
                torch.nn.Tanh(),
            )

        def forward(self, boards):
            boards = boards.permute(0, 3, 1, 2)
            features = self.backbone(boards)
            policy_logits = self.policy_head(features)
            values = self.value_head(features).squeeze(-1)
            return policy_logits, values

    def __init__(self, args: argparse.Namespace):
        # TODO: Define an agent network in `self._model`.
        #
        # A possible architecture known to work consists of
        # - 5 convolutional layers with 3x3 kernel and 15-20 filters,
        # - a policy head, which first uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens the representation, and finally uses a dense layer to produce
        #   the policy logits,
        # - a value head, which again uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens, and produces expected return using an output dense layer with
        #   `tanh` activation.
        
        self.board_shape = (3, 15, 15)
        self.num_actions = 225 # 28 possible actions (fields)

        self._model = self.Model(self.board_shape, self.num_actions).to(self.device)

        self.policy_loss = torch.nn.CrossEntropyLoss()
        self.value_loss = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> "Agent":
        # A static method returning a new Agent loaded from the given path.
        agent = Agent(args)
        agent._model.load_state_dict(torch.load(path, map_location=agent.device))
        return agent

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, boards: torch.Tensor, target_policies: torch.Tensor, target_values: torch.Tensor) -> None:
        # TODO: Train the model based on given boards, target policies and target values.
        # Note that the model returns logits.
        self._model.train()
        
        # Forward pass
        policies, values = self._model(boards)

        # Compute losses
        policy_loss = self.policy_loss(policies, target_policies)
        value_loss = self.value_loss(values, target_values)
        loss = policy_loss + value_loss

        # Backward pass and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, boards: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return the predicted policy and the value function. Because the model
        # returns logits, you should apply softmax to return policy probabilities.

        self._model.eval()  # Set the model to evaluation mode
        with torch.inference_mode():
            policies, values = self._model(boards)
            policy_probs = torch.softmax(policies, dim=1)
        return policy_probs.cpu().numpy(), values.cpu().numpy()
    
    # def board_features(self, game: Pisqorky) -> np.ndarray:
    #     if game.to_play == 0:
    #         prepared_game = game
    #     else:
    #         prepared_game = game.clone(swap_players=True)

    #     return prepared_game.board_features

    def board_features(self, game: Pisqorky) -> np.ndarray:
        board = game.board
        features = np.empty((15, 15, 3), dtype=np.float32)
        features[..., 0] = board == 0
        features[..., 1] = board == 1 + game.to_play
        features[..., 2] = board == 2 - game.to_play
        return features

# ------------
# Augment the board
# ------------
def augment_board_and_policy(board: np.ndarray, policy: np.ndarray):
    policy_board = policy.reshape(Pisqorky.N, Pisqorky.N)

    augmented = []

    for k in range(4):
        aug_board = np.rot90(board, k, axes=(0, 1)).copy()
        aug_policy = np.rot90(policy_board, k).copy().reshape(-1)
        augmented.append((aug_board, aug_policy))

    flipped_board = np.flip(board, axis=1)
    flipped_policy = np.flip(policy_board, axis=1)

    for k in range(4):
        aug_board = np.rot90(flipped_board, k, axes=(0, 1)).copy()
        aug_policy = np.rot90(flipped_policy, k).copy().reshape(-1)
        augmented.append((aug_board, aug_policy))

    return augmented


def train(args: argparse.Namespace) -> Agent:
    # Perform training
    if args.resume:
        print(f"Loading model from {args.model_path}", flush=True)
        agent = Agent.load(args.model_path, args)
    else:
        agent = Agent(args)

    replay_buffer = npfl139.ReplayBuffer(max_length=args.replay_buffer_length)

    def evaluate(boards):
        policies, values = agent.predict(boards)
        return policies, values

    board_game_cpp.simulated_games_start(
        threads=args.threads,
        num_simulations=args.num_simulations,
        sampling_moves=args.sampling_moves,
        epsilon=args.epsilon,
        alpha=args.alpha,
    )

    model_path = prepare_model_path(args)
    # Save experiment config next to the model.
    config_path = model_path.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])

    iteration = 0
    training = True
    best_return = 0

    try:
        while training:
            iteration += 1

            # Generate simulated games
            for _ in range(args.sim_games):
                game = board_game_cpp.simulated_game(evaluate)

                # No rotation
                # entries = [
                #     ReplayBufferEntry(
                #         np.asarray(board, dtype=np.float32),
                #         np.asarray(policy, dtype=np.float32),
                #         np.asarray(value, dtype=np.float32),
                #     )
                #     for board, policy, value in game
                # ]

                # replay_buffer.extend(entries)

                entries = []

                for board, policy, value in game:
                    board = np.asarray(board, dtype=np.float32)
                    policy = np.asarray(policy, dtype=np.float32)
                    value = np.asarray(value, dtype=np.float32)

                    for aug_board, aug_policy in augment_board_and_policy(board, policy):
                        entries.append(
                            ReplayBufferEntry(
                                aug_board,
                                aug_policy,
                                value,
                            )
                        )

                replay_buffer.extend(entries)

            # Train
            for _ in range(args.train_for):
                # TODO: Perform training by sampling an `args.batch_size` of positions
                # from the `replay_buffer` and running `agent.train` on them.
                if len(replay_buffer) < args.batch_size*5:
                    continue
                boards, policies, values = replay_buffer.sample(args.batch_size)
                agent.train(boards, policies, values)

            # Evaluate
            if iteration % args.evaluate_each == 0:
                # Run an evaluation on 2*56 games versus the simple heuristics,
                # using the `Player` instance defined below.
                # For speed, the implementation does not use MCTS during evaluation,
                # but you can of course change it so that it does.
                score = npfl139.board_games.evaluate(
                    Pisqorky, [Player(agent, argparse.Namespace(num_simulations=0)),
                            Pisqorky.player_from_name("random")(seed=main_args.seed)],
                    games=10, first_chosen=False, render=False, verbose=True,
                )
                print(f"Evaluation after iteration {iteration}: {100 * score:.1f}%", flush=True)

                if score >= best_return:
                    best_return = score

                    agent.save(str(model_path))

                    print(
                        f"New best model saved. "
                        f"Score: {100 * best_return:.1f}%, "
                        f"path: {model_path}",
                        flush=True,
                    )

                    if score >= 0.98:
                        print("Target performance reached, stopping training.")
                        training = False
        
        if model_path.exists():
            agent = Agent.load(str(model_path), args)

    finally:
        board_game_cpp.simulated_games_stop()

    return agent


#############################
# BoardGamePlayer interface #
#############################
class Player(npfl139.board_games.BoardGamePlayer[Pisqorky]):
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def evaluate(self, boards):
        policies, values = self.agent.predict(boards)
        return policies, values

    def play(self, game: Pisqorky) -> int:
        # Predict a best possible action.
        if self.args.num_simulations == 0:
            # TODO: If no simulations should be performed, use directly
            # the policy predicted by the agent on the current game board.
            board = self.agent.board_features(game)
            board_batch = board[None]  # Add batch dimension
            policy, _ = self.agent.predict(board_batch)
            policy = policy[0] # Remove batch dimension
        else:
            # TODO: Otherwise run the `mcts` without exploration and
            # utilize the policy returned by it.
            policy = board_game_cpp.mcts(
                game.board,
                game.to_play,
                self.evaluate,
                self.args.num_simulations,
                0.0,
                self.args.alpha,
            )

        # Now select a valid action with the largest probability.
        return max(game.valid_actions(), key=lambda action: policy[action])


########
# Main #
########
def main(args: argparse.Namespace) -> Player:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    board_game_cpp.select_game("pisqorky")

    if args.recodex:
        # Load the trained agent
        agent = Agent.load(args.model_path, args)
    else:
        # Perform training
        agent = train(args)

    return Player(agent, args)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    player = main(main_args)

    # Run an evaluation versus the simple heuristic with the same parameters as in ReCodEx.
    npfl139.board_games.evaluate(
        Pisqorky, [player, Pisqorky.player_from_name("random")(seed=main_args.seed)],
        games=56, first_chosen=False, render=False, verbose=True,
    )

    # My agent againt myself
    # npfl139.board_games.evaluate(
    #     Pisqorky,
    #     [player, Pisqorky.player_from_name("mouse")()],
    #     games=1,
    #     first_chosen=False,
    #     render=True,
    #     verbose=True,
    # )
