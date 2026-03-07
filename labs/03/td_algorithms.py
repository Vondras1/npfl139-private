#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2526.3")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate alpha.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration epsilon factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor gamma.")
parser.add_argument("--mode", default="sarsa", type=str, help="Mode (sarsa/expected_sarsa/tree_backup).")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy; use greedy as target")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=47, type=int, help="Random seed.")
# If you add more arguments, ReCodEx will keep them with your default values.


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a fixed seed.
    generator = np.random.RandomState(args.seed)

    # Create the environment.
    env = npfl139.EvaluationEnv(gym.make("Taxi-v3"), seed=args.seed, report_each=min(200, args.episodes))

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # The next action is always chosen in the epsilon-greedy way.
    def choose_next_action(Q: np.ndarray) -> tuple[int, float]:
        greedy_action = argmax_with_tolerance(Q[next_state])
        next_action = greedy_action if generator.uniform() >= args.epsilon else env.action_space.sample()
        return next_action, args.epsilon / env.action_space.n + (1 - args.epsilon) * (greedy_action == next_action)

    # The target policy is either the behavior policy (if not `args.off_policy`),
    # or the greedy policy (if `args.off_policy`).
    def compute_target_policy(Q: np.ndarray) -> np.ndarray:
        target_policy = np.eye(env.action_space.n)[argmax_with_tolerance(Q, axis=-1)]
        if not args.off_policy:
            target_policy = (1 - args.epsilon) * target_policy + args.epsilon / env.action_space.n
        return target_policy

    # Run the TD algorithm
    for _ in range(args.episodes):
        next_state, done = env.reset()[0], False

        # Generate episode and update Q using the given TD method
        next_action, next_action_prob = choose_next_action(Q)

        # Trajectory buffers:
        # S[t] = state at time t
        # A[t] = action taken at time t
        # R[t+1] = reward observed after action A[t]
        # B[t] = behavior-policy probability of taking A[t] in S[t]
        S = [next_state]
        A = [next_action]
        R = [0.0]
        B = [next_action_prob]

        T = np.inf # Time when episode ends
        t = 0 # Current time step

        # Generate episode and update Q using the given TD method
        # next_action, next_action_prob = choose_next_action(Q)
        while True:
            if t < T:
                action, action_prob, state = A[t], B[t], S[t]
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                S.append(next_state)
                R.append(reward)

                if done:
                    T = t + 1 # Episode ends at time T, so we can update all state-action pairs until T-1
                else:
                    next_action, next_action_prob = choose_next_action(Q)
                    A.append(next_action)
                    B.append(next_action_prob)

            tau = (t + 1) - args.n # Time whose estimate is being updated
            if tau >= 0:
                if args.mode == "sarsa":
                    G = 0
                    end = min(tau + args.n, T)
                    for i in range(tau+1, end+1): # Accumulate rewards until the end of the episode or until tau+n
                        G = G + (args.gamma ** (i - tau - 1)) * R[i] 
                    if tau + args.n < T:
                        G = G + (args.gamma ** args.n) * Q[S[tau + args.n], A[tau + args.n]]
                
                                    # Importance sampling for off-policy SARSA.
                    rho = 1
                    if args.off_policy:
                        target_policy = compute_target_policy(Q)
                        end = min(tau + args.n, T - 1)
                        for i in range(tau + 1, end + 1):
                            target_prob = target_policy[S[i], A[i]]
                            behavior_prob = B[i]
                            rho = rho * target_prob / behavior_prob

                    Q[S[tau], A[tau]] = Q[S[tau], A[tau]] + args.alpha * (G - Q[S[tau], A[tau]]) * rho
                
                elif args.mode == "expected_sarsa":
                    G = 0
                    end = min(tau + args.n, T)
                    for i in range(tau+1, end+1): # Accumulate rewards until the end of the episode or until tau+n
                        G = G + (args.gamma ** (i - tau - 1)) * R[i] 
                    
                    if tau + args.n < T:
                        target_policy = compute_target_policy(Q)
                        expected_value = 0
                        for a in range(env.action_space.n):
                            expected_value = expected_value + target_policy[S[tau + args.n], a] * Q[S[tau + args.n], a]
                        G = G + (args.gamma ** args.n) * expected_value

                    rho = 1
                    if args.off_policy and args.n > 1:
                        target_policy = compute_target_policy(Q)
                        end = min(tau + args.n - 1, T - 1)
                        for i in range(tau + 1, end + 1):
                            target_prob = target_policy[S[i], A[i]]
                            behavior_prob = B[i]
                            rho = rho * target_prob / behavior_prob
                        
                    Q[S[tau], A[tau]] += args.alpha * rho * (G - Q[S[tau], A[tau]])

                elif args.mode == "tree_backup":
                    end = min(tau + args.n, T)
                    if tau + args.n >= T:
                        G = R[T]
                    else:
                        target_policy = compute_target_policy(Q)
                        expected_value = 0
                        for a in range(env.action_space.n):
                            expected_value = expected_value + target_policy[S[end], a] * Q[S[end], a]
                        G = R[end] + args.gamma * expected_value
                    
                    for i in range(end-1, tau, -1):
                        target_policy = compute_target_policy(Q)
                        expected_value = 0
                        for a in range(env.action_space.n):
                            if a == A[i]: # Action taken in the episode
                                expected_value = expected_value + target_policy[S[i], a] * G
                            else:
                                expected_value = expected_value + target_policy[S[i], a] * Q[S[i], a]
                        G = R[i] + args.gamma * expected_value
                    
                    Q[S[tau], A[tau]] += args.alpha * (G - Q[S[tau], A[tau]])

            if tau == T - 1:
                break

            t += 1

            # TODO: Perform the update to the state-action value function `Q`, using
            # a TD update with the following parameters:
            # - `args.n`: use `args.n`-step method
            # - `args.off_policy`:
            #    - if False, the epsilon-greedy behavior policy is also the target policy
            #    - if True, the target policy is the greedy policy
            #      - for SARSA (with any `args.n`) and expected SARSA (with `args.n` > 1),
            #        importance sampling must be used
            # - `args.mode`: this argument can have the following values:
            #   - "sarsa": regular SARSA algorithm
            #   - "expected_sarsa": expected SARSA algorithm
            #   - "tree_backup": tree backup algorithm
            #
            # Perform the updates as soon as you can -- whenever you have all the information
            # to update `Q[state, action]`, do it. For each `action`, use its corresponding
            # `action_prob` from the time of taking the `action` as the behavior policy probability,
            # and the `compute_target_policy(Q)` with the current `Q` (from the time of performing
            # the update) as the target policy.
            #
            # Do not forget that when `done` is True, bootstrapping on the
            # `next_state` is not used.
            #
            # Also note that when the episode ends and `args.n` > 1, there will
            # be several state-action pairs that also need to be updated. Perform
            # the updates in the order in which you encountered the state-action
            # pairs and during these updates, use the `compute_target_policy(Q)`
            # with the up-to-date value of `Q`.

            # if args.mode == "sarsa":
            #     if args.n == 1 and False:
            #         if done:
            #             target = reward
            #         else:
            #             target = reward + args.gamma * Q[next_state, next_action]
            #         Q[state, action] = Q[state, action] + args.alpha * (target - Q[state, action])
            #     # Solve the general case (with n-step and off-policy variants) here.
            #     if done:
            #         target = reward
            #     else:
            #         target = reward + args.gamma * 
            # elif args.mode == "expected_sarsa":
            #     pass # TODO: Implement expected SARSA (with n-step and off-policy variants)
            # elif args.mode == "tree_backup":
            #     pass # TODO: Implement tree backup (with n-step and off-policy variants)



    # Return the final action-value function for ReCodEx to validate.
    return Q


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
