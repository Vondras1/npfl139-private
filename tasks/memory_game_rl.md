### Assignment: memory_game_rl
#### Date: Deadline: Apr 29, 22:00
#### Points: 4 points

This is a continuation of the `memory_game` assignment.

In this task, your goal is to solve the memory game environment
using reinforcement learning. That is, you must not use the
`env.expert_episode` method during training. You can start with PyTorch template
[memory_game_rl.py](https://github.com/ufal/npfl139/tree/master/labs/09/memory_game_rl.py),
which extends the `memory_game` template by generating training episodes
suitable for some reinforcement learning algorithm.

ReCodEx evaluates your solution on environments with 4, 6, and 8 cards (utilizing
the `--cards` argument). In order to pass, your solution must reach
a non-negative average return in 100 evaluation episodes of all the tests.
You can train your agent directly in ReCodEx (the time limit is 5, 10, and 15
minutes for 4, 6, and 8 cards, respectively), or you can submit a pre-trained
one.
