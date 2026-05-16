#!/usr/bin/env python3
import argparse
import collections
import math

from pathlib import Path
import json

import numpy as np
import torch

import npfl139
npfl139.require_version("2526.11.2")
from npfl139.board_games import AZQuiz

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.15, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=256, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=1, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="az_quiz2.pt", type=str, help="Model path")
parser.add_argument("--num_simulations", default=800, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--replay_buffer_length", default=10000, type=int, help="Replay buffer max length.")
parser.add_argument("--sampling_moves", default=10, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=4, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=4, type=int, help="Update steps in every iteration.")

parser.add_argument("--save_dir", default="models", type=str, help="Directory for saved models.")
parser.add_argument("--run_name", default=None, type=str, help="Optional experiment name.")


def prepare_model_path(args: argparse.Namespace) -> Path:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.run_name is None:
        run_name = (
            f"az_quiz"
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

            filters = 20

            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(4, filters, kernel_size=3, padding=1),
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
        
        self.filters = 20
        self.board_shape = (4, 7, 7)
        self.num_actions = 28 # 28 possible actions (fields)

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
        with torch.no_grad():
            policies, values = self._model(boards)
            policy_probs = torch.softmax(policies, dim=1)
        return policy_probs.cpu().numpy(), values.cpu().numpy()

    def board_features(self, game: AZQuiz) -> np.ndarray:
        # TODO: Generate the boards from the current game.
        #
        # The `game.board_features` returns a board representation, but you also
        # need to somehow indicate who is the current player. You can either
        # - change the game so that the current player is always the same one
        #   (i.e., always 0 or always 1; `swap_players` option of `AZQuiz.clone`
        #   method might come handy);
        # - indicate the current player by adding channels to the representation.
        
        # Always represent the board from the perspective of the player to move.
        # After this transformation:
        #   channel 0 = current player's fields
        #   channel 1 = opponent's fields
        if game.to_play == 0:
            prepared_game = game
        else:
            prepared_game = game.clone(swap_players=True)
        
        return prepared_game.board_features

def get_outcome(game, to_play):
    outcome = game.outcome(to_play)
    if outcome is None:
        return None
    else:
        return outcome.value-2

########
# MCTS #
########
class MCTNode:
    def __init__(self, prior: float | None):
        self.prior = prior  # Prior probability from the agent.
        self.game = None    # If the node is evaluated, the corresponding game instance.
        self.children = {}  # If the node is evaluated, mapping of valid actions to the child `MCTNode`s.
        self.visit_count = 0
        self.total_value = 0

    def value(self) -> float:
        # TODO: Return the value of the current node, handling the
        # case when `self.visit_count` is 0.
        if self.visit_count == 0:
            # TODO: Try to set the value according to the parent node.
            # However, for now lets go just with 0
            return 0
        else:
            return self.total_value/self.visit_count

    def is_evaluated(self) -> bool:
        # A node is evaluated if it has non-zero `self.visit_count`.
        # In such case `self.game` is not None.
        return self.visit_count > 0

    def evaluate(self, game: AZQuiz, agent: Agent) -> None:
        # Each node can be evaluated at most once
        assert self.game is None
        self.game = game

        # TODO: Compute the value of the current game.
        # - If the game has ended, compute the value directly
        # - Otherwise, use the given `agent` to evaluate the current
        #   game. Then, for all valid actions, populate `self.children` with
        #   new `MCTNodes` with the priors from the policy predicted
        #   by the network.
        # outcome = game.outcome(game.to_play)
        outcome = get_outcome(game, game.to_play)
        if outcome is not None:
            value = outcome
        else:
            board = agent.board_features(game)
            board_batch = board[None]  # Add batch dimension
            policy, value = agent.predict(board_batch)
            policy = policy[0]
            value = value[0]

            # Normalize priors for valid actions!
            valid_actions = game.valid_actions()
            valid_priors = np.array([policy[a] for a in valid_actions])
            sum_priors = valid_priors.sum()
            
            if sum_priors > 0:
                valid_priors /= sum_priors
            else:
                valid_priors = np.ones(len(valid_actions)) / len(valid_actions)


            for i, action in enumerate(valid_actions):
                self.children[action] = MCTNode(valid_priors[i])

        self.total_value += value
        self.visit_count += 1

    def add_exploration_noise(self, epsilon: float, alpha: float) -> None:
        # TODO: Update the children priors by exploration noise
        # Dirichlet(alpha), so that the resulting priors are
        #   epsilon * Dirichlet(alpha) + (1 - epsilon) * original_prior
        actions = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))

        for action, noise_value in zip(actions, noise):
            child = self.children[action] # V aktuálním uzlu projdi přes všechny jeho děti a aktualizuj prior o exploring noice
            child.prior = epsilon * noise_value + (1 - epsilon) * child.prior

    def select_child(self) -> tuple[int, "MCTNode"]:
        # Select a child according to the PUCT formula.
        def ucb_score(child: "MCTNode"):
            # TODO: For a given child, compute the UCB score as
            #   Q(s, a) + C(s) * P(s, a) * (sqrt(N(s)) / (N(s, a) + 1)),
            # where:
            # - Q(s, a) is the estimated value of the action stored in the
            #   `child` node. However, the value in the `child` node is estimated
            #   from the view of the player playing in the `child` node, which
            #   is usually the other player than the one playing in `self`,
            #   and in that case the estimated value must be "inverted";
            # - C(s) in AlphaZero is defined as
            #     log((1 + N(s) + 19652) / 19652) + 1.25
            #   Personally I used 1965.2 to account for shorter games, but I do not
            #   think it makes any difference;
            # - P(s, a) is the prior computed by the agent;
            # - N(s) is the number of visits of state `s`;
            # - N(s, a) is the number of visits of action `a` in state `s`.
            
            Q = -child.value()  # Invert the value because the child is from the perspective of the opponent
            C = math.log((1+self.visit_count+1965.2)/1965.2) + 1.25
            H = math.sqrt(self.visit_count) / (child.visit_count + 1)
            ucb_score = Q + C * child.prior * H
            return ucb_score

        # TODO: Return the (action, child) pair with the highest `ucb_score`.
        best_action = None
        best_child = None
        best_score = -np.inf

        for action, child in self.children.items():
            score = ucb_score(child)

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child


def mcts(game: AZQuiz, agent: Agent, args: argparse.Namespace, explore: bool) -> np.ndarray:
    # Run the MCTS search and return the policy proportional to the visit counts,
    # optionally including exploration noise to the root children.
    root = MCTNode(None)
    root.evaluate(game, agent)
    if explore:
        root.add_exploration_noise(args.epsilon, args.alpha)

    # Perform the `args.num_simulations` number of MCTS simulations.
    for _ in range(args.num_simulations):
        # TODO: Starting in the root node, traverse the tree using `select_child()`,
        # until a `node` without `children` is found.
        node = root
        search_path = [root]
        action_path = []

        # Go through the tree until an unevaluated node or terminal node is found
        while node.children:
            action, child = node.select_child()
            node = child
            search_path.append(node)
            action_path.append(action)

        # If the node has not been evaluated, evaluate it.
        if not node.is_evaluated():
            # TODO: Evaluate the `node` using the `evaluate` method. To that
            # end, create a suitable `AZQuiz` instance for this node by cloning
            # the `game` from its parent and performing a suitable action.
            if len(search_path) == 1:
                # If we are at the root, we can directly use the given game.
                node_game = game
            else:
                parent = search_path[-2]
                action = action_path[-1]
                node_game = parent.game.clone()
                node_game.move(action)

            node.evaluate(node_game, agent)

            # Get the value of the node.
            value = node.value()

        else:
            # TODO: If the node has been evaluated but has no children, the
            # game ends in this node. Update it appropriately.
            # value = node.game.outcome(node.game.to_play).value
            value = get_outcome(node.game, node.game.to_play)
            node.total_value += value
            node.visit_count += 1

        # TODO: For all parents of the `node`, update their value estimate,
        # i.e., the `visit_count` and `total_value`.
        for parent in reversed(search_path[:-1]):
            value = -value
            parent.visit_count += 1
            parent.total_value += value

    # TODO: Compute a policy proportional to visit counts of the root children.
    # Note that invalid actions are not the children of the root, but the
    # policy should still return 0 for them.
    policy = np.zeros(28, dtype=np.float32)

    for action, child in root.children.items():
        policy[action] = child.visit_count
    
    # Normalize the policy to sum to 1
    if policy.sum() > 0:
        policy /= policy.sum()
    else:
        policy[:] = 1 / len(policy) # Should never happen (I think)
    return policy


############
# Training #
############
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])


def sim_game(agent: Agent, args: argparse.Namespace) -> list[ReplayBufferEntry]:
    # Simulate a game, return a list of `ReplayBufferEntry`s.
    game = AZQuiz()
    history = []
    # while not game.outcome(game.to_play):
    while get_outcome(game, game.to_play) == None:

        # TODO: Run the `mcts` with exploration.
        policy = mcts(game, agent, args, explore=True)

        history.append((agent.board_features(game), policy, game.to_play))

        # TODO: Select an action, either by sampling from the policy or greedily,
        # according to the `args.sampling_moves`.
        if len(history) <= args.sampling_moves:
            action = np.random.choice(len(policy), p=policy)
        else:
            action = np.argmax(policy)

        game.move(action)
    
    # final_outcome = game.outcome(0).value  # Outcome from the perspective of player 0
    final_outcome = get_outcome(game, 0)
    # print("f = ", final_outcome)

    # TODO: Return all encountered game states, each consisting of
    # - the board (probably via `agent.board_features`),
    # - the policy obtained by MCTS,
    # - the outcome based on the outcome of the whole game.
    replay = []
    for board, policy, player in history:
        if player == 0:
            outcome = final_outcome
        else:
            outcome = -final_outcome
        replay.append(ReplayBufferEntry(board, policy, outcome))
    
    return replay


def train(args: argparse.Namespace) -> Agent:
    # Perform training
    agent = Agent(args)
    replay_buffer = npfl139.ReplayBuffer(max_length=args.replay_buffer_length)

    model_path = prepare_model_path(args)

    # Save experiment config next to the model.
    config_path = model_path.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    iteration = 0
    training = True
    best_return = 0
    while training:
        iteration += 1

        # Generate simulated games
        for _ in range(args.sim_games):
            game = sim_game(agent, args)
            replay_buffer.extend(game)

            # If required, show the generated game, as 8 very long lines showing
            # all encountered boards, each field showing as
            # - `XX` for the fields belonging to player 0,
            # - `..` for the fields belonging to player 1,
            # - percentage of visit counts for valid actions.
            if args.show_sim_games:
                log = [[] for _ in range(8)]
                for i, (board, policy, outcome) in enumerate(game):
                    log[0].append(f"Move {i}, result {outcome}".center(28))
                    action = 0
                    for row in range(7):
                        log[1 + row].append("  " * (6 - row))
                        for col in range(row + 1):
                            log[1 + row].append(
                                " XX " if board[row, col, 0] else
                                " .. " if board[row, col, 1] else
                                f"{policy[action] * 100:>3.0f} ")
                            action += 1
                        log[1 + row].append("  " * (6 - row))
                print(*["".join(line) for line in log], sep="\n")

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
                AZQuiz, [Player(agent, argparse.Namespace(num_simulations=0)),
                         AZQuiz.player_from_name("simple_heuristic")(seed=main_args.seed)],
                games=56, first_chosen=False, render=False, verbose=False,
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

                # if score >= 0.98:
                #     print("Target performance reached, stopping training.")
                #     training = False

    return agent


#############################
# BoardGamePlayer interface #
#############################
class Player(npfl139.board_games.BoardGamePlayer[AZQuiz]):
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: AZQuiz) -> int:
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
            policy = mcts(game, self.agent, self.args, explore=False)

        # Now select a valid action with the largest probability.
        return max(game.valid_actions(), key=lambda action: policy[action])


########
# Main #
########
def main(args: argparse.Namespace) -> Player:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

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
        AZQuiz, [player, AZQuiz.player_from_name("simple_heuristic")(seed=main_args.seed)],
        games=56, first_chosen=False, render=False, verbose=True,
    )
