#!/usr/bin/env python3
# Team:
# 1ac5d633-f96f-42a3-846d-31bcb01d041f
# e0cfa255-0259-11eb-9574-ea7484399335
# 9fafb47f-e1c5-4d7c-8ce5-8a6f5bdcd751

import argparse

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import npfl139
npfl139.require_version("2526.5")

import torch
import numpy as np
import collections



parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--frame_stack", default=4, type=int, help="Frame stack.")
parser.add_argument("--game", default="Pong", type=str, help="Game to play.")
parser.add_argument("--grayscale", default=True, action=argparse.BooleanOptionalAction, help="Grayscale obs.")
parser.add_argument("--screen_size", default=84, type=int, help="Screen size.")

parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=100, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=1000, type=int, help="Target update frequency.")
parser.add_argument("--warmup", default=2000, type=int, help="Warmup - buffer capacity.")
parser.add_argument("--warm_start", default=True, action="store_true", help="Running with warm start from MODEL_PATH_INIT.")


parser.add_argument("--grad_max_norm", default=10.0, type=float, help="Gradient clipping norm")

parser.add_argument("--atoms", default=51, type=int, help="Number of atoms.")
parser.add_argument("--v_min", default=-30.0, type=float, help="Minimum support value.")
parser.add_argument("--v_max", default=30.0, type=float, help="Maximum support value.")


MODEL_PATH = "atari_double_c51.pt"
MODEL_PATH_INIT = "atari_double_c51_5p2_init.pt"


class Network:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        n_actions = int(env.action_space.n)
        n_atoms   = int(args.atoms)
        self._grad_max_norm = args.grad_max_norm
        self._gamma = args.gamma

        # CNN part of the newtwork
        in_channels = args.frame_stack if args.grayscale else 3 * args.frame_stack

        self._cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )

        obs_shape = env.observation_space.shape

        if len(obs_shape) == 3:
            # Could be CHW or HWC
            if obs_shape[0] == in_channels:
                dummy = torch.zeros(1, *obs_shape)
            else:
                dummy = torch.zeros(1, obs_shape[2], obs_shape[0], obs_shape[1])
        else:
            raise ValueError(f"Unexpected observation shape: {obs_shape}")

        with torch.no_grad():
            cnn_out_size = self._cnn(dummy).view(1, -1).shape[1]    

        # Dense part of the network
        self._model = torch.nn.Sequential(
            self._cnn,
            torch.nn.Flatten(),
            torch.nn.Linear(cnn_out_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, n_actions * n_atoms),
            torch.nn.Unflatten(1, (n_actions, n_atoms)),
        ).to(self.device)

        self._model.register_buffer(
            "atoms", torch.linspace(args.v_min, args.v_max, args.atoms, device=self.device)
        )

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

    def _prepare_states(self, states: torch.Tensor) -> torch.Tensor:
        states = states.float()
        # If in is NHWC, converts to NCHW.
        if states.ndim == 4 and states.shape[-1] in [1, 3, 4, 12]:
            states = states.permute(0, 3, 1, 2)
        return states / 255.0

    @staticmethod
    def compute_loss(
        states_logits: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states_online_logits: torch.Tensor,
        next_states_target_logits: torch.Tensor,
        atoms: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:

        batch_size = states_logits.shape[0]
        n_atoms = atoms.shape[0]
        batch_idx = torch.arange(batch_size, device=states_logits.device)

        chosen_logits = states_logits[batch_idx, actions] 

        with torch.no_grad():
    
            next_online_probs = torch.softmax(next_states_online_logits.detach(), dim=-1)   
            next_online_q = torch.sum(next_online_probs * atoms, dim=-1)                    
            next_actions = torch.argmax(next_online_q, dim=1)                              

            next_target_probs = torch.softmax(next_states_target_logits.detach(), dim=-1)   
            p_next = next_target_probs[batch_idx, next_actions]                              

            v_min, v_max = atoms[0], atoms[-1]
            delta_z = atoms[1] - atoms[0]

            tz = rewards[:, None] + gamma * (1.0 - dones[:, None]) * atoms[None, :]
            tz = torch.clamp(tz, v_min, v_max)

            b = (tz - v_min) / delta_z
            l = torch.floor(b).long()
            u = torch.ceil(b).long()

            l = torch.clamp(l, 0, n_atoms - 1)
            u = torch.clamp(u, 0, n_atoms - 1)

            target_dist = torch.zeros(batch_size, n_atoms, device=states_logits.device)

            target_dist.scatter_add_(1, l, p_next * (u.float() - b))
            target_dist.scatter_add_(1, u, p_next * (b - l.float()))

            eq_mask = (l == u)
            target_dist.scatter_add_(1, l, p_next * eq_mask.float())

        log_probs = torch.log_softmax(chosen_logits, dim=-1)
        loss = -(target_dist * log_probs).sum(dim=1).mean()
        return loss

    @npfl139.typed_torch_function(
        device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32, torch.float32
    )
    def train(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states_online_logits: torch.Tensor,
        next_states_target_logits: torch.Tensor,
    ) -> None:
        self._model.train()

        states = self._prepare_states(states)
        states_logits = self._model(states)

        loss = self.compute_loss(
            states_logits,
            actions,
            rewards,
            dones,
            next_states_online_logits,
            next_states_target_logits,
            self._model.atoms,
            self._gamma,
        )

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=self._grad_max_norm)
        self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            states = self._prepare_states(states)
            logits = self._model(states)                    # [B, A, N]
            probs = torch.softmax(logits, dim=-1)          # [B, A, N]
            q_values = torch.sum(probs * self._model.atoms, dim=-1)  # [B, A]
            return q_values

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_logits(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            states = self._prepare_states(states)
            return self._model(states)

    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> None:
        self._model.load_state_dict(torch.load(path, map_location=self.device))



def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=args.frame_skip, grayscale_obs=args.grayscale, screen_size=args.screen_size)
    env = gym.wrappers.FrameStackObservation(env, stack_size=args.frame_stack)



    network = Network(env, args)
    if args.warm_start:
        print("Loading model from " + MODEL_PATH_INIT + ".")    
        network.load(MODEL_PATH_INIT)   
    
    target_network = Network(env, args)
    target_network.copy_weights_from(network)

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent
        network.load(MODEL_PATH)
        # Final evaluation
        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False
            while not done:
                # TODO: Choose a greedy action
                q_values = network.predict(state[np.newaxis])[0]
                action = int(np.argmax(q_values))
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # TODO: Train an agent using for example some distributed-RL algorithm.
    #
    # If you want to create N multithreaded parallel environments, use
    #   vector_env = ale_py.AtariVectorEnv(
    #       game=re.sub(r"(?<=[a-z])(?=[A-Z])", "_", args.game).lower(),  # use snake_case for the game name
    #       num_envs=N,  # the requred number of parallel environments,
    #       frameskip=args.frame_skip, stack_num=args.frame_stack, grayscale=args.grayscale,
    #       img_height=args.screen_size, img_width=args.screen_size,
    #       use_fire_reset=False, reward_clipping=False, repeat_action_probability=0.25,
    #       autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    #   )
    #
    # There are several Autoreset modes available, see https://farama.org/Vector-Autoreset-Mode.
    # In some situations, the SAME_STEP might be more practical than the default NEXT_STEP mode.
    
    
    replay_buffer = npfl139.ReplayBuffer(max_length=100_000)
    Transition = collections.namedtuple(
        "Transition", ["state", "action", "reward", "done", "next_state"]
    )

    step = 0
    epsilon = args.epsilon
    returns_best = -np.inf
    training = True
    episode = 0

    while training:
        state, done = env.reset()[0], False
        episode += 1

        while not done:

            q_values = network.predict(state[np.newaxis])[0]
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = int(np.argmax(q_values))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.append(Transition(state, action, reward, done, next_state))
            step += 1

            if len(replay_buffer) >= args.warmup:
                batch = replay_buffer.sample(args.batch_size)

                next_online_logits = network.predict_logits(batch.next_state)
                next_target_logits = target_network.predict_logits(batch.next_state)

                network.train(
                    batch.state,
                    batch.action,
                    batch.reward,
                    batch.done.astype(np.float32),
                    next_online_logits,
                    next_target_logits,
                )

                if step % args.target_update_freq == 0:
                    target_network.copy_weights_from(network)

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(
                episode,
                [0, args.epsilon_final_at],
                [args.epsilon, args.epsilon_final],
            )

        avg_return = np.mean(main_env._episode_returns[-10:]) if main_env._episode_returns else -np.inf
        if avg_return > returns_best:
            returns_best = avg_return
            network.save(MODEL_PATH)
            print(f"Episode {episode}, avg return {avg_return:.2f}, epsilon {epsilon:.3f} -> saved")

        if returns_best >= 16.5:
            training = False


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    assert main_args.render_each in [0, 1], "Option render_each can be only 0 or 1 for Atari games"

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make(f"ALE/{main_args.game}-v5", frameskip=1, render_mode="human" if main_args.render_each else None),
        main_args.seed)

    main(main_env, main_args)
