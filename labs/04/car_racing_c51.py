#!/usr/bin/env python3
# Team:
# 1ac5d633-f96f-42a3-846d-31bcb01d041f
# 9fafb47f-e1c5-4d7c-8ce5-8a6f5bdcd751

import argparse
import collections
import os
import re
import json

import gymnasium as gym
import numpy as np
import torch

from gymnasium.wrappers.vector import ResizeObservation as VectorResizeObservation
from gymnasium.wrappers.vector import GrayscaleObservation as VectorGrayscaleObservation

import npfl139
npfl139.require_version("2526.4")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--continuous", default=1, type=int, help="Use continuous actions.")
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=1000, type=int, help="Target update frequency.")
parser.add_argument("--evaluation_episodes", default=50, type=int, help="Number of evaluation episodes.")
parser.add_argument("--num_envs", default=8, type=int, help="Number of parallel environments.")
parser.add_argument("--max_episodes", default=1000, type=int, help="Maximum number of episodes.")
parser.add_argument("--atoms", default=100, type=int, help="Number of atoms.")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actions = [
    [-0.2, 0.0, 0.0],   # gentle steer left
    [-0.8, 0.0, 0.0],   # strong steer left
    [ 0.2, 0.0, 0.0],   # gentle steer right
    [ 0.8, 0.0, 0.0],   # strong steer right
    [ 0.0, 0.4, 0.0],   # light throttle
    [ 0.0, 1.0, 0.0],   # full throttle
    [ 0.0, 0.0, 0.4],   # light braking
    [ 0.0, 0.0, 0.8],   # strong braking
]
actions_n = len(actions)

class AgentSaver:
    def __init__(self, args, main_folder, model_name = "q_model_racing.pt"):
        self.args = args
        self.recodex = args.recodex
        self.main_folder = main_folder
        self.model_name = model_name
        self.args_name = "args.json"

    def save_next_agent(self, model) -> None:
        """Find the highest numbered agent folder in base_folder and save the new agent as +1."""
        os.makedirs(self.main_folder, exist_ok=True)

        pattern = re.compile(r"agent_(\d+)")
        max_id = 0

        for name in os.listdir(self.main_folder):
            match = pattern.fullmatch(name)
            if match:
                agent_id = int(match.group(1))
                max_id = max(max_id, agent_id)

        new_id = max_id + 1
        new_folder = os.path.join(self.main_folder, f"agent_{new_id}")

        self.save_agent(new_folder, model)

        print(f"Agent saved to {new_folder}")

    def save_agent(self, folder_path: str, model) -> None:
        """Save model weights and parser arguments into one folder."""
        os.makedirs(folder_path, exist_ok=True)

        model_path = os.path.join(folder_path, self.model_name)
        args_path = os.path.join(folder_path, self.args_name)

        torch.save(model.state_dict(), model_path)

        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, indent=4, ensure_ascii=False)

    def load_agent(self, model, agent_name: str) -> tuple[object, dict | None]:
        """Load trained model and saved parser arguments from one folder."""

        if self.recodex:
            model_path = "q_model_racing.pt"
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            return model, None

        model_path = os.path.join(self.main_folder, agent_name, self.model_name)
        args_path = os.path.join(self.main_folder, agent_name, self.args_name)

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        with open(args_path, "r", encoding="utf-8") as f:
            saved_args = json.load(f)

        return model, saved_args

def choose_action(env, epsilon, q_values, actions_n):
    if np.random.rand() < epsilon:
        action = np.random.randint(0, actions_n)
    else:
        action = np.argmax(q_values)
    return action

def make_stacked_state(frame_buffer):
    return np.concatenate(list(frame_buffer), axis=-1)

def evaluate_training(eval_env, model, args, target_value = 500):
    returns = []
    for _ in range(args.evaluation_episodes):
        state, done = eval_env.reset()[0], False

        frame_buffer = collections.deque(maxlen=4)
        for _ in range(4):
            frame_buffer.append(state)
        stacked_state = make_stacked_state(frame_buffer)

        episode_return = 0
        while not done:
            # Choose a greedy action
            action = np.argmax(model.predict(stacked_state[np.newaxis])[0])
            state, reward, terminated, truncated, _ = eval_env.step(actions[action])
            done = terminated or truncated
            episode_return += reward

            frame_buffer.append(state)
            stacked_state = make_stacked_state(frame_buffer)

        returns.append(episode_return)

    mean_return = np.mean(returns)
    print("Evaluation return:", mean_return)
    if mean_return > target_value:
        print("Target reached, stopping training.")
        return True, mean_return

    return False, mean_return

class Network(torch.nn.Module):
    def __init__(self, env, args, actions_n):
        super().__init__()
        self.env = env
        self.args = args

        # action_num = env.action_space.n
        H, W, C = env.observation_space.shape

        # Create `self._model.atoms` as uniform grid from 0 to 500 with `args.atoms` elements.
        # We create them as a buffer in `self._model` so they are automatically moved with `.to`.
        self.register_buffer("atoms", torch.linspace(-200, 1000, args.atoms))

        in_channels = C * 4 #(4 stacked images to gain access to speed)
        out_channels = 32
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = torch.nn.ReLU()
        # 32 x 32 x 32

        in_channels = out_channels
        out_channels = 64
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        # 16 x 16 x 64

        in_channels = out_channels
        out_channels = 128
        self.conv3 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        # 4 x 4 x 128

        # in_channels = out_channels
        # out_channels = 256
        # self.conv4 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        # self.batch_norm4 = torch.nn.BatchNorm2d(out_channels)
        # self.relu = torch.nn.ReLU()
        # 4 x 4 x 256

        # pool
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))

        # Linear
        self.fc1 = torch.nn.Linear(out_channels, 256)
        self.fc2 = torch.nn.Linear(256, actions_n*args.atoms)
        self.unflattern = torch.nn.Unflatten(1, (int(actions_n), int(args.atoms)))

        self._optimizer = torch.optim.AdamW(self.parameters(), lr=args.learning_rate)
        self._loss = torch.nn.MSELoss()
        self.to(DEVICE)


    def forward(self, x):
        # Input: (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2).float() / 255.0

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        # (N, 128, 4, 4)
        x = self.pool(x)
        # (N, 128, 1, 1)
        x = torch.flatten(x, 1)
        # (N, 128)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.unflattern(x)
        return x
    

    @staticmethod
    def compute_loss(
        states_logits: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
        next_states_logits_online: torch.Tensor, next_states_logits_target: torch.Tensor, atoms: torch.Tensor, gamma: float,
    ) -> torch.Tensor:
        # TODO: Implement the loss computation according to the C51 algorithm.
        # - The `states_logits` are current state logits of shape `[batch_size, actions, atoms]`.
        # - The `actions` are the integral actions taken in the states, of shape `[batch_size]`.
        # - The `rewards` are the rewards obtained after taking the actions, of shape `[batch_size]`.
        # - The `dones` are `torch.float32` indicating whether the episode ended, of shape `[batch_size]`.
        # - The `next_states_logits_online` are logits of the next states, of shape `[batch_size, actions, atoms]`.
        #   Because they should not be backpropagated through, use an appropriate `.detach()` call.
        # - The `next_states_logits_target` are logits of the target next states, of shape `[batch_size, actions, atoms]`.
        #   (obtained by target network), use an appropriate `.detach()` call.
        # - The `atoms` are the atom values. Your implementation must handle any number of atoms. The
        #   `atoms[0]` is V_MIN (the minimum atom value), `atoms[-1]` is V_MAX (the maximum atom value),
        #   and use `atoms[1] - atoms[0]` as the distance between two consecutive atoms. You can
        #   assume that one of the atoms is always 0.
        # The resulting loss should be the mean of the cross-entropy losses of the individual batch examples.
        #
        # Your implementation most likely needs to be vectorized to pass ReCodEx time limits. Note that you
        # can add given values to a vector of (possibly repeating) tensor indices using `scatter_add_`.
        
        batch_size = actions.shape[0]
        batch_indices = torch.arange(batch_size, device=actions.device)

        # [batch, # of atoms], for every sample one distribution
        train_logits = states_logits[batch_indices, actions]

        V_min = atoms[0]
        V_max = atoms[-1]
        delta_z = atoms[1] - atoms[0]

        # Get probabilities from next state logits
        next_probs_online = torch.softmax(next_states_logits_online.detach(), dim=-1)   # [B, A, N]
        next_probs_target = torch.softmax(next_states_logits_target.detach(), dim=-1)   # [B, A, N]
        # Compute Q values from each ditribution (One value for every action). From ONLINE network
        Qs_next = (next_probs_online*atoms).sum(dim=-1)
        # Choose the best action
        greedy_actions = torch.argmax(Qs_next, dim=1)
        # From target network take the distribution of the greedy action in the next state
        greedy_actions_distribution = next_probs_target[batch_indices, greedy_actions]

        # shift the next-state atom support by reward and discount
        tz = rewards[:, None] + gamma * (1 - dones[:, None]) * atoms[None, :]
        tz = torch.clip(tz, V_min, V_max)

        # project shifted atom values onto the fixed atom grid
        b = (tz - V_min)/delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        # Target distribution
        m = torch.zeros_like(greedy_actions_distribution)   # [B, N]

        # ditribute the probability # HAHAHAHAHAHAHAHA
        offset = (torch.arange(batch_size, device=actions.device) * atoms.shape[0])[:, None]
        offset = offset.expand_as(l)

        m.view(-1).scatter_add_(
            0,
            (l + offset).reshape(-1),
            (greedy_actions_distribution * (u.float() - b)).reshape(-1)
        )
        m.view(-1).scatter_add_(
            0,
            (u + offset).reshape(-1),
            (greedy_actions_distribution * (b - l.float())).reshape(-1)
        )
        eq_mask = (l == u)
        m.view(-1).scatter_add_(
            0,
            (l[eq_mask] + offset[eq_mask]).reshape(-1),
            greedy_actions_distribution[eq_mask].reshape(-1)
        )

        # Finall cross-entropy loss between the predicted probabilities `log_probs` and the target distribution `m`
        log_probs = torch.log_softmax(train_logits, dim=-1)
        loss = -(m * log_probs).sum(dim=1).mean()
        return loss


    # # Define a training method. Generally you have two possibilities
    # # - pass new q_values of all actions for a given state; all but one are the same as before
    # # - pass only one new q_value for a given state, and include the index of the action to which
    # #   the new q_value belongs
    # # The code below implements the first option, but you can change it if you want.
    # #
    # # The `npfl139.typed_torch_function` automatically converts input arguments
    # # to PyTorch tensors of given type, and converts the result to a NumPy array.
    # @npfl139.typed_torch_function(DEVICE, torch.float32, torch.float32)
    # def train_step(self, states: torch.Tensor, q_values: torch.Tensor) -> None:
    #     self.train()
    #     predictions = self(states)
    #     loss = self._loss(predictions, q_values)

    #     self._optimizer.zero_grad()
    #     loss.backward()
    #     with torch.no_grad():
    #         self._optimizer.step()

    # The training function defers the computation to the `compute_loss` method.
    def train_step(self, states, actions, rewards, dones, next_states, target_model) -> None:
        super().train()

        states = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=DEVICE)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=DEVICE)

        states_logits = self(states)
        next_states_logits_online = self(next_states)

        with torch.no_grad():
            next_states_logits_target = target_model(next_states)

        loss = self.compute_loss(
            states_logits,
            actions,
            rewards,
            dones,
            next_states_logits_online,
            next_states_logits_target,
            self.atoms,
            self.args.gamma,
        )

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self._optimizer.step()

    # @npfl139.typed_torch_function(DEVICE, torch.float32)
    # def predict(self, states: torch.Tensor) -> np.ndarray:
    #     self.eval()
    #     with torch.no_grad():
    #         return self(states)
    @npfl139.typed_torch_function(DEVICE, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            # TODO: Return all predicted Q-values for the given states.
            logits = self(states)                      # [B, A, N]
            probs = torch.softmax(logits, dim=-1)             # [B, A, N]
            q_values = (probs * self.atoms).sum(dim=-1)  # [B, A]
            return q_values

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: "Network") -> None:
        self.load_state_dict(other.state_dict())

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Create evaluation environment
    eval_env = npfl139.EvaluationEnv(
        gym.make("npfl139/CarRacingFS-v3", frame_skip=args.frame_skip, continuous=args.continuous),
        args.seed, args.render_each, evaluate_for=15, report_each=1)
    
    # If you want, you can wrap even the `npfl139.EvaluationEnv` with additional wrappers, like
    #   env = gym.wrappers.ResizeObservation(env, (64, 64))
    # or
    #   env = gym.wrappers.GrayscaleObservation(env)
    # However, if you do that, you can no longer call just `env.reset(start_evaluation=True)`;
    # instead, you need to pass the `start_evaluation` to the inner environment using
    #   env.reset(options={"start_evaluation": True})
    env = gym.wrappers.ResizeObservation(env, (64, 64))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)

    eval_env = gym.wrappers.ResizeObservation(eval_env, (64, 64))
    eval_env = gym.wrappers.GrayscaleObservation(eval_env, keep_dim=True)

    # Construct the online network
    network = Network(env, args, actions_n)

    # Construct the target network
    target_network = Network(env, args, actions_n)
    target_network.copy_weights_from(network)

    # Replay memory; the `max_length` parameter is its maximum capacity.
    replay_buffer = npfl139.ReplayBuffer(max_length=1_000_000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon

    # Helper for saving/loading agents
    agent = AgentSaver(args, "racing_c51", model_name = "q_model_racing.pt")

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent
        model, _ = agent.load_agent(network, None)

        # Final evaluation
        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False

            frame_buffer = collections.deque(maxlen=4)
            for _ in range(4):
                frame_buffer.append(state)
            stacked_state = make_stacked_state(frame_buffer)

            while not done:
                # TODO: Choose a greedy action
                action = np.argmax(model.predict(stacked_state[np.newaxis])[0])
                state, reward, terminated, truncated, _ = env.step(actions[action])
                done = terminated or truncated
                frame_buffer.append(state)
                stacked_state = make_stacked_state(frame_buffer)

    # TODO: Implement a suitable RL algorithm and train the agent.
    #
    # If you want to create N multiprocessing parallel environments, use
    #   vector_env = gym.make_vec("npfl139/CarRacingFS-v3", N, gym.VectorizeMode.ASYNC,
    #                             frame_skip=args.frame_skip, continuous=args.continuous)
    #   vector_env.reset(seed=args.seed)  # The individual environments get incremental seeds
    #
    # There are several Autoreset modes available, see https://farama.org/Vector-Autoreset-Mode.
    # To change the autoreset mode to SAME_STEP from the default NEXT_STEP, pass
    #   vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
    # as an additional argument to the above `gym.make_vec`.
    training = True
    train_steps = 0
    best_return = 0
    episode = 0
    while training:
        # Perform episode
        state, done = env.reset()[0], False

        frame_buffer = collections.deque(maxlen=4)
        for _ in range(4):
            frame_buffer.append(state)
        stacked_state = make_stacked_state(frame_buffer)

        while not done:
            # Epsilon-greedy action selection.
            q_values = network.predict(stacked_state[np.newaxis])[0]
            action = choose_action(env, epsilon, q_values, actions_n)

            # Make a step
            next_state, reward, terminated, truncated, _ = env.step(actions[action])
            done = terminated or truncated

            frame_buffer.append(next_state)
            stacked_next_state = make_stacked_state(frame_buffer)

            # Append state, action, reward, done and next_state to replay_buffer
            # replay_buffer.append(Transition(state, action, reward, done, next_state))
            replay_buffer.append(Transition(stacked_state, action, reward, done, stacked_next_state))

            if len(replay_buffer) > args.batch_size*30:
                samples = replay_buffer.sample(args.batch_size, replace=True)

                # Double C51:
                # - online network selects next action
                # - target network provides next-state distribution
                network.train_step(
                    samples.state,
                    samples.action,
                    samples.reward,
                    samples.done,
                    samples.next_state,
                    target_network,
                )

                # # ------ Double Deep Q Network ------
                # q_values = network.predict(samples.state)

                # next_q_online = network.predict(samples.next_state)
                # next_actions = np.argmax(next_q_online, axis=1)

                # next_q_target = target_network.predict(samples.next_state)
                # next_q_selected = next_q_target[np.arange(args.batch_size), next_actions]

                # targets = samples.reward + args.gamma * next_q_selected * (~samples.done)

                # q_values[np.arange(args.batch_size), samples.action] = targets
                # network.train_step(samples.state, q_values)

                train_steps += 1
                if train_steps % args.target_update_freq == 0:
                    print("Weights updated")
                    target_network.copy_weights_from(network)

            state = next_state
            stacked_state = stacked_next_state

        episode += 1

        # Evaluate regularly and stop once the target is reached.
        if episode % 50 == 0:
            target_reached, mean_return = evaluate_training(eval_env, network, args, target_value = 900)
            if ((mean_return > 750 and best_return < mean_return) or target_reached):
                agent.save_next_agent(network)
                print(f"Mean return = {mean_return}")
                best_return = mean_return
            if (target_reached or args.max_episodes < episode): 
                break # Finish training if target reached or max allowed number of episodes exceeded


        if args.epsilon_final_at:
            epsilon = np.interp(episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

    # Save model
    agent.save_next_agent(network)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("npfl139/CarRacingFS-v3", frame_skip=main_args.frame_skip, continuous=main_args.continuous),
        main_args.seed, main_args.render_each, evaluate_for=15, report_each=1)

    main(main_env, main_args)
